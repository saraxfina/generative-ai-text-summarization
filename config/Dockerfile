FROM python:3.10

WORKDIR /app

COPY ./config/requirements.txt /app/requirements.txt 
COPY ./models /models 

COPY ./src/server.py /app/server.py

RUN apt update 
RUN apt-get install -y libhdf5-serial-dev
RUN pip install --no-binary=h5py h5py
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


