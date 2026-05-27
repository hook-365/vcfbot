FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The package lives under src/vcfbot; mount it at /app/vcfbot so the import
# path matches `vcfbot.server:app` without needing pip-install.
COPY src/vcfbot ./vcfbot
ENV PYTHONPATH=/app

# Drop privileges. uid 1000 matches the typical host user (anthony on
# apollo), so anything chromadb writes to the bind-mounted ./data/ ends
# up host-owned, not root-owned. Avoids the "rsync into chroma fails
# with Permission denied" trap entirely.
RUN groupadd -g 1000 vcfbot \
 && useradd -u 1000 -g 1000 -m -s /usr/sbin/nologin vcfbot \
 && chown -R vcfbot:vcfbot /app
USER vcfbot

EXPOSE 8129

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8129/api/status >/dev/null || exit 1

CMD ["uvicorn", "vcfbot.server:app", \
     "--host", "0.0.0.0", "--port", "8129", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
