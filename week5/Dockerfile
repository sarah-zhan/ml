FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

# WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_flask_q6.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--listen=0.0.0.0:9696", "predict_flask_q6:app"]