import subprocess
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration du benchmark
SCRIPTS = {
    "Serial": {"file": "game_of_life.py", "mpi": False},
    "Lignes (1D)": {"file": "linesplit.py", "mpi": True},
    "Colonnes (1D)": {"file": "colsplit.py", "mpi": True},
    "Blocs (2D)": {"file": "2Dsplit.py", "mpi": True},
    
}

PROCESS_COUNTS = [1, 2, 4, 8]  # Nombre de processus MPI à tester
PATTERN = "glider"         # Motif à simuler
RESOLUTION = ["800", "800"]    # Résolution de la grille
RUN_DURATION = 5               # Temps d'exécution par test (en secondes)

def parse_output(stdout_data, is_serial):
    """Extrait les temps de calcul et d'affichage depuis la sortie standard."""
    calc_times = []
    rend_times = []
    
    # Séparation par retours chariot \r ou \n
    lines = re.split(r'[\r\n]+', stdout_data.decode('utf-8', errors='ignore'))
    
    for line in lines:
        if is_serial:
            # Match: Temps calcul prochaine generation : 1.23e-02 secondes, temps affichage : ...
            match = re.search(r'calcul.*?: ([\d\.e\-\+]+).*?affichage.*?: ([\d\.e\-\+]+)', line)
        else:
            # Match: Calc/Calcul/Workers: 1.23e-02s | Rendu/Master: 1.23e-02s
            match = re.search(r': ([\d\.e\-\+]+)s.*? (?:Rendu|Master|Maître).*?: ([\d\.e\-\+]+)s', line)
            
        if match:
            calc_times.append(float(match.group(1)))
            rend_times.append(float(match.group(2)))
            
    if not calc_times:
        return np.nan, np.nan
        
    # Ignorer la première itération (souvent faussée par l'initialisation)
    if len(calc_times) > 1:
        calc_times, rend_times = calc_times[1:], rend_times[1:]
        
    return np.mean(calc_times), np.mean(rend_times)

def run_benchmark():
    results = []
    
    # 1. Test de la version sérielle (référence)
    print("--- Lancement du test Sérial ---")
    cmd = ["python", "-u", SCRIPTS["Serial"]["file"], PATTERN] + RESOLUTION
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(RUN_DURATION)
    proc.terminate()
    stdout, _ = proc.communicate()
    
    calc_t, rend_t = parse_output(stdout, True)
    serial_total = calc_t + rend_t
    results.append({
        "Stratégie": "Serial",
        "Processus": 1,
        "Calcul (s)": calc_t,
        "Affichage (s)": rend_t,
        "Total (s)": serial_total,
        "Speedup": 1.0
    })
    
    # 2. Tests des versions MPI
    for name, config in SCRIPTS.items():
        if not config["mpi"]:
            continue
            
        for p in PROCESS_COUNTS:
            print(f"--- Lancement de {name} avec {p} processus ---")
            cmd = ["mpiexec", "-n", str(p), "python", "-u", config["file"], PATTERN] + RESOLUTION
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(RUN_DURATION)
            proc.terminate()
            stdout, _ = proc.communicate()
            
            calc_t, rend_t = parse_output(stdout, False)
            total_t = calc_t + rend_t
            
            # Calcul du Speedup : T_serial / T_parallel
            speedup = serial_total / total_t if not np.isnan(total_t) and total_t > 0 else np.nan
            
            results.append({
                "Stratégie": name,
                "Processus": p,
                "Calcul (s)": calc_t,
                "Affichage (s)": rend_t,
                "Total (s)": total_t,
                "Speedup": speedup
            })
            
    return pd.DataFrame(results)

def generate_plots(df):
    """Génère les 4 graphiques demandés."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse des Performances - Conway's Game of Life (MPI)", fontsize=16)
    
    strategies = df["Stratégie"].unique()
    markers = ['o', 's', '^', 'D']
    
    ax_total = axs[0, 0]
    ax_calc = axs[0, 1]
    ax_aff = axs[1, 0]
    ax_speed = axs[1, 1]
    
    for idx, strat in enumerate(strategies):
        data = df[df["Stratégie"] == strat]
        
        # Ligne de base sérielle (affichée comme ligne horizontale pour référence si désiré)
        if strat == "Serial":
            for ax, col in zip([ax_total, ax_calc, ax_aff], ["Total (s)", "Calcul (s)", "Affichage (s)"]):
                ax.axhline(y=data[col].values[0], color='gray', linestyle='--', label='Serial (1 proc)')
            continue
            
        ax_total.plot(data["Processus"], data["Total (s)"], marker=markers[idx], label=strat)
        ax_calc.plot(data["Processus"], data["Calcul (s)"], marker=markers[idx], label=strat)
        ax_aff.plot(data["Processus"], data["Affichage (s)"], marker=markers[idx], label=strat)
        ax_speed.plot(data["Processus"], data["Speedup"], marker=markers[idx], label=strat)

    # Configuration des axes
    plots_info = [
        (ax_total, "Temps Total Moyen par Génération", "Temps (s)"),
        (ax_calc, "Temps de Calcul Moyen par Génération", "Temps (s)"),
        (ax_aff, "Temps d'Affichage Moyen (Rank 0)", "Temps (s)"),
        (ax_speed, "Speedup Global ($T_{serial} / T_{parallel}$)", "Speedup")
    ]
    
    for ax, title, ylabel in plots_info:
        ax.set_title(title)
        ax.set_xlabel("Nombre de processus")
        ax.set_ylabel(ylabel)
        ax.set_xticks(PROCESS_COUNTS)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        
    # Ajouter la ligne de speedup idéal (y = x)
    ax_speed.plot(PROCESS_COUNTS, PROCESS_COUNTS, 'k--', alpha=0.5, label='Speedup Idéal')
    ax_speed.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Démarrage du benchmark (cela va ouvrir/fermer des fenêtres Pygame)...")
    df_results = run_benchmark()
    
    print("\n" + "="*50)
    print("TABLEAU RÉSUMÉ DES PERFORMANCES")
    print("="*50)
    print(df_results.to_markdown(index=False, floatfmt=".4f"))
    
    print("\nGénération des graphiques...")
    generate_plots(df_results)
