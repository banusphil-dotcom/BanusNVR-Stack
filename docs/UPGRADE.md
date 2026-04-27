# Upgrading

## Standard upgrade

```bash
cd BanusNVR-Stack
git pull
docker compose pull
docker compose up -d
docker image prune -f
```

The database schema migrates automatically on `api` startup. Watch logs:

```bash
docker compose logs -f api
```

## Pinning a version

To pin to a specific release (recommended for production):

```env
# .env
BANUSNVR_TAG=v1.2.3
```

Then `docker compose pull && docker compose up -d`.

## Rolling back

```env
BANUSNVR_TAG=v1.2.2
```

Then `docker compose pull && docker compose up -d`.

If a migration was applied that the older version doesn't know about, you may
need to restore a database backup first.

## Backing up before upgrading

```bash
# Postgres
docker compose exec db pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" \
  | gzip > backups/db-$(date +%F).sql.gz

# Frigate config
cp config/frigate/config.yml backups/frigate-config-$(date +%F).yml
```

## Major-version notes

Breaking changes are listed in `CHANGELOG.md` and tagged on GitHub releases.
Always read those before bumping a major version.
