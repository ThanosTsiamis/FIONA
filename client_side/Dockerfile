# Build stage: install locked dependencies and produce a production build.
FROM node:20.18.0-bookworm-slim AS builder

WORKDIR /usr/src/app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# NEXT_PUBLIC_* values are inlined into the client bundle at build time, so the
# API URL must be provided as a build argument, not only as a runtime variable.
ARG NEXT_PUBLIC_API_URL=http://localhost:5000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build && npm prune --omit=dev

# Runtime stage: only the compiled app and production dependencies.
FROM node:20.18.0-bookworm-slim AS runner

WORKDIR /usr/src/app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN groupadd --system --gid 10001 nodejs \
    && useradd --system --uid 10001 --gid nodejs --no-create-home --home-dir /usr/src/app nextjs

COPY --from=builder --chown=nextjs:nodejs /usr/src/app/package.json ./package.json
COPY --from=builder --chown=nextjs:nodejs /usr/src/app/next.config.js ./next.config.js
COPY --from=builder --chown=nextjs:nodejs /usr/src/app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /usr/src/app/public ./public
COPY --from=builder --chown=nextjs:nodejs /usr/src/app/node_modules ./node_modules

USER nextjs

EXPOSE 3000

CMD ["node_modules/.bin/next", "start", "-H", "0.0.0.0"]
