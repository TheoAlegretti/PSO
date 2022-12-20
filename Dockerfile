FROM python:3

# Créer un répertoire de travail
WORKDIR /app

# Copier les fichiers requis dans le répertoire de travail
COPY requirements.txt .
COPY app.py .

# Installer les dépendances
RUN pip install -r requirements.txt

# Exposer le port 5000
EXPOSE 5000

# Définir l'environnement d'exécution
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0

# Exécuter l'application lorsque le conteneur est lancé
CMD ["flask", "run"]



