import matplotlib.pyplot as plt

def plot_translation_curves():
    shifts = [0, 8, 16, 32]
    
    resnet_acc = [95.80, 95.35, 94.30, 93.70]
    vit_acc = [97.80, 97.45, 97.35, 96.75]
    clip_acc = [97.20, 95.85, 95.75, 94.25]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(shifts, resnet_acc, marker='o', label='ResNet-50', linewidth=2)
    ax.plot(shifts, vit_acc, marker='s', label='ViT-B/16', linewidth=2)
    ax.plot(shifts, clip_acc, marker='^', label='CLIP ViT-B-32', linewidth=2)
    
    ax.set_xticks(shifts)
    ax.set_xlabel('Displacement (Pixels)')
    ax.set_ylabel('Target Accuracy (%)')
    ax.set_title('Translation Invariance Degradation')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.savefig('task1/results/translation_curve_accuracy.png', dpi=300)
    print("Translation accuracy curve saved to task1/results/translation_curve_accuracy.png")

if __name__ == "__main__":
    plot_translation_curves()
