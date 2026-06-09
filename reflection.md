# Project Reflection

## What Worked Well

- The CNN architecture was straightforward to build and reached the 99% target quickly.
- Adding Dropout(0.25) was the single most impactful change — it cleanly reduced the train/val gap.
- The experiment table (one change per run) made it easy to see what each modification contributed.
- Data augmentation gave a small but consistent accuracy boost.

## What Did Not Work as Expected

- Early runs without Dropout overfit noticeably: training accuracy was ~99.5% while validation was only 98.7%.
- Increasing epochs beyond 10 gave diminishing returns and slightly increased loss variance.

## What I Would Change

- Start with Dropout from Run 1 to save experiment time.
- Use a learning-rate scheduler (ReduceLROnPlateau) for smoother convergence in longer runs.
- Save a checkpoint per epoch and restore the best weights automatically.

## Next Steps

1. Test the model on photos of my own handwriting.
2. Build a simple web interface (Gradio or Streamlit) for live digit prediction.
3. Retrain on EMNIST for letter recognition.
4. Explore knowledge distillation to produce a smaller, faster model for mobile deployment.
