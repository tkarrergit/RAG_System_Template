import logging

def setup_logging():
    logging.basicConfig(
        filename="rag_system.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

# Initialisiere das Logging
setup_logging()

# Erstelle einen Logger für das gesamte Projekt
logger = logging.getLogger(__name__)
logger.info("Das Logging wurde initialisiert.")