# Използваме официален slim Python образ
FROM python:3.10-slim

# Обновяваме apt и инсталираме системни зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Подготвяме pip
RUN pip install --upgrade pip setuptools wheel

# Копираме requirements и инсталираме Python зависимости (вкл. FAISS)
COPY requirements.txt .
RUN pip install --no-cache-dir pip-tools && \
    pip-sync requirements.txt

# Копираме цялото приложение в контейнера
WORKDIR /app
COPY . .

# Отваряме порт, ако е Flask
EXPOSE 5000

# Стартова команда
CMD ["python", "app.py"]