FROM python:3.12-slim
WORKDIR /app
COPY fit_toolkit.py .
EXPOSE 5050
CMD ["python3", "fit_toolkit.py"]
