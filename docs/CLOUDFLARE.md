# Cloudflare Tunnel

Cloudflare Tunnel lets you expose BanusNVR securely on the public internet
**without opening any ports** on your router. Your origin server makes an
outbound TLS connection to Cloudflare and traffic flows through it.

## 1. Create the tunnel

1. Sign in to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/).
2. **Networks → Tunnels → Create a tunnel**.
3. Choose **Cloudflared**, name it (e.g. `banusnvr`).
4. **Save tunnel** → copy the long token shown on the install page (do not
   bother with the install command — Compose runs `cloudflared` for you).

## 2. Add a public hostname

In the **Public Hostname** tab, click **Add a public hostname**:

- **Subdomain**: `nvr`
- **Domain**: your domain on Cloudflare
- **Service type**: `HTTP`
- **URL**: `web:80`

Save. (You can add additional hostnames pointing to `frigate:5000` for the
Frigate UI if you wish.)

## 3. Wire the token into BanusNVR

Set in your `.env`:

```env
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoi…   # paste here
```

Restart the stack:

```bash
docker compose up -d
```

`cloudflared` will start automatically because the token is now non-empty.

## 4. Lock it down (recommended)

Back in Zero Trust, **Access → Applications → Add an application →
Self-hosted**, target the public hostname, and add a policy that requires
e.g. an email OTP or your Google account. This puts a Cloudflare login wall
in front of BanusNVR's own login.

## Disabling

Comment out or empty `CLOUDFLARE_TUNNEL_TOKEN` and run
`docker compose up -d` to recreate without the tunnel.
