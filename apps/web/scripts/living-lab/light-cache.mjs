// Read-only local research assets. Bind localhost; no remote writes or directory listing.
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve, sep } from "node:path";
const root = resolve(process.env.LIGHT_CACHE);
const server = createServer(async (req, res) => {
  const path = resolve(root, "." + decodeURIComponent(new URL(req.url, "http://local").pathname));
  res.setHeader("Access-Control-Allow-Origin", "*");
  if ((req.method !== "GET" && req.method !== "HEAD") || !path.startsWith(root + sep)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const info = await stat(path);
    if (!info.isFile()) throw new Error("not a file");
    res.setHeader("Content-Length", info.size);
    res.setHeader(
      "Content-Type",
      path.endsWith("json") ? "application/json" : "application/octet-stream",
    );
    if (req.method === "HEAD") res.end();
    else createReadStream(path).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
});
server.listen(Number(process.env.LIGHT_CACHE_PORT ?? 5602), "127.0.0.1");
