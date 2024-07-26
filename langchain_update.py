import toml
import subprocess

def update_dependencies(prefix):
    # Carica il file pyproject.toml
    with open('pyproject.toml', 'r') as file:
        pyproject = toml.load(file)
    
    # Estrai le dipendenze dalla sezione [tool.poetry.dependencies]
    dependencies = pyproject['tool']['poetry']['dependencies']
    
    # Filtra le dipendenze che iniziano con il prefisso specificato
    deps_to_update = [dep for dep in dependencies if dep.startswith(prefix)]
    
    # Esegui il comando poetry update per ciascuna di queste dipendenze
    for dep in deps_to_update:
        subprocess.run(['poetry', 'update', dep])

# Aggiorna tutte le dipendenze che iniziano con "langchain"
update_dependencies('langchain')
