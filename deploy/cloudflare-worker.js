// Cloudflare Worker for the ResearchReaderNerd domain.
//
// Proxies every request to the ECS Express service hostname and adds the shared
// origin secret. The app refuses any request without it (except /health), so the
// AWS hostname can't be used to skip Cloudflare Access and its login.
//
// Set in the Cloudflare dashboard (Workers & Pages > this Worker > Settings):
//   Variable ORIGIN:          the Express hostname, e.g. researchreadernerd-xxxx.ecs.us-east-1.on.aws
//   Secret   ORIGIN_SECRET:   the same value stored in AWS as researchreadernerd/origin-secret
// Route: <your domain>/*  (with a Cloudflare Access application on the same domain)

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    url.hostname = env.ORIGIN;
    url.protocol = "https:";
    url.port = "";
    const proxied = new Request(url, request);
    proxied.headers.set("X-Origin-Secret", env.ORIGIN_SECRET);
    return fetch(proxied);
  },
};
