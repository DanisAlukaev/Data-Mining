FROM python
COPY . .
RUN pip install --no-cache-dir pandas && \
    pip install -r requirements.txt && \
    chmod +x main.py
WORKDIR .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5432"]