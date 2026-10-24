# ResearchReaderNerd: AWS deployment

Same shape as ProtocolNerd: one container on **ECS Fargate** (Express Mode), images in
**ECR**, GitHub Actions with OIDC for build and deploy, in the same AWS account
(921135845455) but with its own roles, bucket, repository and secrets.

**Current choice (2026-10-24):** only you have access, at the AWS hostname
(`https://<service>.ecs.us-east-1.on.aws`), behind a password. Cloudflare in front, with
a custom domain and email login, is optional and can be added later (step 8) without
changing the app.

```
browser ── HTTPS ──> ECS Express hostname  <service>.ecs.us-east-1.on.aws
                       └─> Fargate task: rrn-web on :8080  (asks for the password)
                             ├─ reports, uploads, run records ─> S3 bucket (private)
                             └─ ANTHROPIC_API_KEY, password ─> Secrets Manager
```

Later, with Cloudflare (step 8):

```
your domain
  └─> Cloudflare Access (login: only listed emails)
        └─> Cloudflare Worker (deploy/cloudflare-worker.js), adds X-Origin-Secret
              └─> ECS Express hostname  <service>.ecs.us-east-1.on.aws
                    └─> Fargate task: rrn-web on :8080
                          ├─ reports, uploads, run records ─> S3 bucket (private)
                          └─ ANTHROPIC_API_KEY, RRN_ORIGIN_SECRET ─> Secrets Manager
```

Why each piece:

| Piece | Reason |
|---|---|
| Password (`RRN_BASIC_AUTH_PASSWORD`) | The AWS hostname is public. Every analysis spends API credit, and the prompt panel lets a visitor rewrite the system prompt, so nothing but `/health` answers without the password. |
| Origin secret (for Cloudflare, later) | Lets requests through when they come from your Cloudflare Worker after a Cloudflare Access login. Either the password or this header is enough. |
| S3 | Fargate's disk is wiped on every restart and deploy. Reports and uploads must outlive the task. |
| One task | Runs in progress live in the task's memory. Run records are in S3, so a deploy never loses a finished report. |
| `RRN_DAILY_RUN_LIMIT` | A ceiling on spend even if a login leaks. Set to 50 below; change as you like. |

## Settings

Run everything from the repository root, with AWS CLI v2 credentials for the target account.

```bash
export AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
APP=researchreadernerd
BUCKET=$APP-reports-$ACCOUNT
CLUSTER=default            # Express Mode's default cluster
```

## One-time setup

### 1. ECR repository

```bash
aws ecr create-repository --repository-name $APP --image-scanning-configuration scanOnPush=true
aws ecr put-lifecycle-policy --repository-name $APP --lifecycle-policy-text file://deploy/ecr-lifecycle-policy.json
```

### 2. S3 bucket (private, encrypted)

```bash
aws s3api create-bucket --bucket $BUCKET --region us-east-1
aws s3api put-public-access-block --bucket $BUCKET --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket $BUCKET --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

### 3. Secrets

A separate Anthropic key for this app is recommended, so its spend is visible on its
own and it can be revoked without touching ProtocolNerd. The commands below read the
key from `.env` without printing it.

```bash
aws secretsmanager create-secret --name $APP/anthropic-api-key \
  --secret-string "$(grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2- | tr -d \"\'\")"
aws secretsmanager create-secret --name $APP/password \
  --secret-string "$(openssl rand -base64 18)"
aws secretsmanager create-secret --name $APP/origin-secret \
  --secret-string "$(openssl rand -hex 32)"
```

To see the password (you need it to sign in; any username works):

```bash
aws secretsmanager get-secret-value --secret-id $APP/password --query SecretString --output text
```

The origin secret is only needed if you add Cloudflare later (step 8).

### 4. IAM roles

```bash
# Task execution role: pulls the image, writes logs, reads this app's secrets.
aws iam create-role --role-name ${APP}ExecutionRole --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ${APP}ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name ${APP}ExecutionRole --policy-name read-secrets \
  --policy-document "$(sed "s/ACCOUNT_ID/$ACCOUNT/g" deploy/execution-role-secrets-policy.json)"

# Task role: what the app itself may do. Only read/write its own bucket.
aws iam create-role --role-name ${APP}TaskRole --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name ${APP}TaskRole --policy-name reports-bucket \
  --policy-document "$(sed "s/BUCKET_NAME/$BUCKET/g" deploy/task-role-s3-policy.json)"

# Infrastructure role for Express Mode. Skip if it already exists (ProtocolNerd may have one):
aws iam get-role --role-name ecsInfrastructureRoleForExpressServices >/dev/null 2>&1 || {
  aws iam create-role --role-name ecsInfrastructureRoleForExpressServices --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
}
```

IAM changes take about a minute to propagate. If step 6 fails with an assume-role
error, wait and retry.

### 5. First image

Later images are built by GitHub Actions; the first one is pushed by hand so the
service has something to run.

```bash
REPO=$ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/$APP
SHA=$(git rev-parse --short HEAD)
aws ecr get-login-password | docker login --username AWS --password-stdin ${REPO%%/*}
docker build --platform linux/amd64 --build-arg BUILD_SHA=$SHA \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) -t $REPO:$SHA .
docker push $REPO:$SHA
```

### 6. Express Mode service

`--cpu` and `--memory` are in CPU units and MiB per the CLI reference (512 = 0.5 vCPU,
1024 = 1 GB). AWS's tutorial shows a different unit style, so check with
`--generate-cli-skeleton input` if the call is rejected.

```bash
SECRETS_ARN_PREFIX=arn:aws:secretsmanager:us-east-1:$ACCOUNT:secret:$APP
aws ecs create-express-gateway-service \
  --service-name $APP \
  --cluster $CLUSTER \
  --execution-role-arn arn:aws:iam::$ACCOUNT:role/${APP}ExecutionRole \
  --task-role-arn arn:aws:iam::$ACCOUNT:role/${APP}TaskRole \
  --infrastructure-role-arn arn:aws:iam::$ACCOUNT:role/ecsInfrastructureRoleForExpressServices \
  --cpu 512 --memory 1024 \
  --health-check-path /health \
  --scaling-target minTaskCount=1,maxTaskCount=1 \
  --primary-container "{
    \"image\": \"$REPO:$SHA\",
    \"containerPort\": 8080,
    \"environment\": [
      {\"name\": \"RRN_S3_BUCKET\", \"value\": \"$BUCKET\"},
      {\"name\": \"RRN_DAILY_RUN_LIMIT\", \"value\": \"50\"}
    ],
    \"secrets\": [
      {\"name\": \"ANTHROPIC_API_KEY\", \"valueFrom\": \"$(aws secretsmanager describe-secret --secret-id $APP/anthropic-api-key --query ARN --output text)\"},
      {\"name\": \"RRN_BASIC_AUTH_PASSWORD\", \"valueFrom\": \"$(aws secretsmanager describe-secret --secret-id $APP/password --query ARN --output text)\"},
      {\"name\": \"RRN_ORIGIN_SECRET\", \"valueFrom\": \"$(aws secretsmanager describe-secret --secret-id $APP/origin-secret --query ARN --output text)\"}
    ]
  }" \
  --monitor-resources
```

When it reports ACTIVE, note the service URL (`https://<name>.ecs.us-east-1.on.aws`) and
check it:

```bash
curl -s https://<name>.ecs.us-east-1.on.aws/health   # build sha, "storage": "S3Storage"
curl -s -o /dev/null -w '%{http_code}\n' https://<name>.ecs.us-east-1.on.aws/   # 401: password required
```

Find the task-definition family Express Mode created (needed by the deploy workflow):

```bash
aws ecs describe-services --cluster $CLUSTER --services $APP \
  --query 'services[0].taskDefinition' --output text    # .../task-definition/<FAMILY>:<REV>
```

### 7. GitHub Actions role and variables

If the account has no GitHub OIDC provider yet (ProtocolNerd's setup created one):

```bash
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com
```

```bash
aws iam create-role --role-name ${APP}GithubDeploy \
  --assume-role-policy-document "$(sed "s/ACCOUNT_ID/$ACCOUNT/g" deploy/github-oidc-trust-policy.json)"
aws iam put-role-policy --role-name ${APP}GithubDeploy --policy-name build-and-deploy \
  --policy-document "$(sed -e "s/ACCOUNT_ID/$ACCOUNT/g" -e "s/CLUSTER_NAME/$CLUSTER/g" deploy/github-deploy-policy.json)"
```

In GitHub, Settings > Secrets and variables > Actions > Variables, add:

| Variable | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::<ACCOUNT>:role/researchreadernerdGithubDeploy` |
| `ECR_REPOSITORY` | `researchreadernerd` |
| `ECS_CLUSTER` | `default` |
| `ECS_SERVICE` | `researchreadernerd` |
| `ECS_TASK_FAMILY` | the family from step 6 |
| `HEALTH_URL` | `https://<name>.ecs.us-east-1.on.aws/health` |

Then open `https://<name>.ecs.us-east-1.on.aws/` in a browser and sign in with any
username and the password from step 3.

### 8. Cloudflare (optional, later)

Adds a custom domain and email login in front. The password keeps working alongside it.

1. **Worker:** create one from `deploy/cloudflare-worker.js`. Add a variable `ORIGIN` (the
   Express hostname, without `https://`) and a secret `ORIGIN_SECRET` (the value of
   `researchreadernerd/origin-secret`). Route it to `<your domain>/*`.
2. **DNS:** a proxied record for the domain (the Worker handles the traffic).
3. **Access:** Zero Trust > Access > Applications > Self-hosted, on the same domain,
   with a policy allowing only the emails you choose (one-time PIN login is enough).

Then `https://<your domain>/` asks for a login, and after it shows the app.

## Deploying a change

1. Push to `main`. "Build and push image" runs the tests, then pushes the image tagged with the short SHA.
2. Run "Deploy to production" with that SHA. It rolls out, smoke-tests `/health` (build SHA
   and S3 storage), and rolls back on failure.

## Rolling back

Run "Deploy to production" with an older SHA, or point the service at the previous
task-definition revision:

```bash
aws ecs update-service --cluster default --service researchreadernerd \
  --task-definition <FAMILY>:<PREVIOUS_REV>
```

## Cost

Rough, not checked against current AWS prices: a 0.5 vCPU / 1 GB task running all
month is around $15–20, plus the load balancer Express Mode creates. Claude usage is
separate: about $0.15–0.25 per paper with Sonnet 5.5 at high effort. The daily run
limit caps the worst case at 50 analyses a day.

## Removing everything

```bash
aws ecs delete-express-gateway-service --service-arn <service ARN>   # also removes its load balancer
# then, if wanted: the S3 bucket (after emptying it), the ECR repository, the two secrets,
# the three researchreadernerd* IAM roles, and the Cloudflare Worker and Access app.
```
