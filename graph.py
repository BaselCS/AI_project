import os
import json
import matplotlib.pyplot as plt

from const import OLD_DIR, SAVE_DIR


def plot_best_fitness_per_generation(new_name=None):
    if new_name is None:
        new_name = SAVE_DIR
        
    if not os.path.exists(new_name):
        print("لا توجد بيانات محفوظة.")
        return
    
    generation_id = []
    fitnesses = []
    dino_id = []
    scores = []

    for folder in sorted(os.listdir(new_name), key=lambda x: int(x.split('_')[1])):
        gen_path = os.path.join(new_name, folder, "generation_data.json")
        if os.path.isfile(gen_path):
            with open(gen_path, 'r') as f:
                data = json.load(f)
                generation_id.append(data['generation'])
                fitnesses.append(data['highest_fitness'])
                dino_id.append(data['highest_fitness_genome_id'])
                scores.append(data['score'])

    if not generation_id:
        print("لم يتم العثور على أي بيانات محفوظة.")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot both metrics on the same graph
    line1, = plt.plot(generation_id, fitnesses, 'b-o', label='Fitness')
    line2, = plt.plot(generation_id, scores, 'r-s', label='Score')
    

    plt.xlabel("gen")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    
    # Find peak points for both metrics
    max_fit_idx = fitnesses.index(max(fitnesses))
    max_score_idx = scores.index(max(scores))
    
    # Annotate peak fitness
    plt.annotate(f'gen: {generation_id[max_fit_idx]}\n fitness: {fitnesses[max_fit_idx]:.2f} at dino: {dino_id[max_score_idx]}',
                xy=(generation_id[max_fit_idx], fitnesses[max_fit_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Annotate peak score
    plt.annotate(f'gen: {generation_id[max_score_idx]}\n score: {scores[max_score_idx]:.2f} at dino: {dino_id[max_score_idx]}',
                xy=(generation_id[max_score_idx], scores[max_score_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if new_name==None:
        plt.show()
    else:
        plt.savefig(new_name)

if __name__ == "__main__":
    
    # for dic in os.listdir(OLD_DIR):
    #     if os.path.isdir(os.path.join(OLD_DIR, dic)):
    #         print(f"Processing directory: {dic}")
            #   plot_best_fitness_per_generation(os.path.join(OLD_DIR, dic))
    plot_best_fitness_per_generation()