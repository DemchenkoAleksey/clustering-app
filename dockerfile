# Multi-stage build
FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.12-slim

WORKDIR /app

# Копируем установленные пакеты из builder stage
COPY --from=builder /root/.local /root/.local

# Копируем код приложения
COPY . .

# Делаем sure что скрипты в PATH
ENV PATH=/root/.local/bin:$PATH

# Создаем non-root пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]