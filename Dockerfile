FROM python:3.13-alpine
COPY . /CalHsePred
WORKDIR /CalHsePred
RUN pip install -r requirements.txt
CMD python app.py
