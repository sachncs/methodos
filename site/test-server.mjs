import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";

const port = 4325;
const root = resolve("dist");
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = createServer(async (request, response) => {
  const requestPath = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
  const relativePath = requestPath.startsWith("/methodos/")
    ? requestPath.slice("/methodos/".length)
    : requestPath.replace(/^\/+/, "");
  let candidate = normalize(join(root, relativePath || "index.html"));
  if (existsSync(candidate) && statSync(candidate).isDirectory()) {
    candidate = join(candidate, "index.html");
  }
  if (
    (candidate !== root && !candidate.startsWith(`${root}/`)) ||
    !existsSync(candidate) ||
    !statSync(candidate).isFile()
  ) {
    response.writeHead(404).end("Not found");
    return;
  }
  response.writeHead(200, { "content-type": contentTypes[extname(candidate)] ?? "application/octet-stream" });
  createReadStream(candidate).pipe(response);
});

server.listen(port, "127.0.0.1");
