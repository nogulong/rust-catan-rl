import subprocess
from importlib import resources

pkg_root = resources.files("pycatan")

def start_server(port):
    cmd = [
        "java",
        "-Djsettlers.forceStartSeat=0",
        "-jar", f"{pkg_root}/JSettlersServer-2.7.00.jar",
        "-Djsettlers.startrobots=12",
        "-Djsettlers.startrobots.fast=6",
        "-Djsettlers.bots.cookie=cookie",
        str(port),
        "100"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def open_gui(port):
    cmd = [
        "java",
        "-jar", f"{pkg_root}/JSettlers-2.7.00.jar",
        "localhost",
        str(port)
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)