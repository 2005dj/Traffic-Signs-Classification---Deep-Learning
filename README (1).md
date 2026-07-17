# Image Classification of Road Signs

A simple project that trains a computer to look at a photo and tell whether it contains a **stop sign** or **not**.

## What it does

The notebook takes a set of labeled road images, learns from them, and then can look at a brand-new photo and predict: *stop sign* or *no stop sign*.

Instead of teaching the model everything from zero, it starts from a model that already knows how to recognize shapes and objects (called **transfer learning**) and just teaches it the one new task. This means it learns fast and needs only a small number of images.

## How it works

1. **Get the data** — the notebook automatically downloads the image set.
2. **Prepare the images** — resizes them to a standard size and splits them into a group for learning and a group for testing.
3. **Train the model** — the model studies the training images.
4. **Check the results** — it measures how often it gets the test images right.
5. **Try it out** — you can feed in your own image and see what it predicts.

## How to run

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Open `Image_Classification_of_Road_Signs.ipynb` and run the cells from top to bottom.

The dataset downloads on its own, so there's nothing to set up manually. Training takes only a few minutes.

## Notes

- The trained model is saved as `model.pt` when you run the notebook, so it doesn't need to be included here — it's recreated each time.
- The downloaded images and result folders are also generated when you run it.

## Built with

Python, PyTorch, and a few common image and math libraries.
