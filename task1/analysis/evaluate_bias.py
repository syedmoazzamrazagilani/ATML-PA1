import numpy as np

def calculate_shape_bias(n_shape, n_texture, n_total):
    """Calculates Shape Bias(%) and Coverage(%) as defined in the manual."""
    if (n_shape + n_texture) == 0:
        return 0.0, 0.0
    
    shape_bias = (n_shape / (n_shape + n_texture)) * 100
    coverage = ((n_shape + n_texture) / n_total) * 100
    return shape_bias, coverage

def calculate_translation_consistency(clean_preds, translated_preds):
    """Calculates Consistency(delta) as the fraction of unchanged predictions."""
    clean_preds = np.array(clean_preds)
    translated_preds = np.array(translated_preds)
    
    if len(clean_preds) != len(translated_preds):
        raise ValueError("Prediction arrays must be the same length.")
        
    matches = (clean_preds == translated_preds).astype(int)
    consistency = np.mean(matches)
    return consistency

if __name__ == "__main__":
    # Example usage for report generation
    n_shape_resnet, n_texture_resnet, n_total = 44, 156, 200
    bias, cov = calculate_shape_bias(n_shape_resnet, n_texture_resnet, n_total)
    print(f"ResNet-50 -> Shape Bias: {bias:.2f}%, Coverage: {cov:.2f}%")
