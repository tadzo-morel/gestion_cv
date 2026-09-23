"""
Script pour vérifier que tout est bien installé
"""

print("Vérification de l'installation...\n")

# Vérifier les bibliothèques
libraries = [
    ('numpy', 'numpy'), ('pandas', 'pandas'), ('sklearn', 'scikit-learn'),
    ('matplotlib', 'matplotlib'), ('seaborn', 'seaborn'), ('wordcloud', 'wordcloud'),
    ('flask', 'flask'), ('flask_cors', 'flask-cors'),
    ('pypdf', 'pypdf'), ('docx', 'python-docx'), ('requests', 'requests'),
]

missing = []
for lib, pip_name in libraries:
    try:
        __import__(lib)
        print(f"✓ {lib}")
    except ImportError:
        print(f"✗ {lib} - MANQUANT")
        missing.append(pip_name)

if missing:
    print(f"\n❌ Bibliothèques manquantes: {', '.join(missing)}")
    print("\nInstalle-les avec:")
    print(f"pip install {' '.join(missing)}")
    print("(ou simplement : pip install -r requirements.txt)")
else:
    print("\n✅ Toutes les bibliothèques sont installées!")
    print("Tu peux commencer à utiliser le système.")