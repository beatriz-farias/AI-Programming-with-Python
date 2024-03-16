import argparse
import utils

parser = argparse.ArgumentParser()

# Adding the arguments to the parser
parser.add_argument('image_path', action='store', default= 'flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', action='store', default= 'checkpoint.pt')
parser.add_argument('--top_k', action='store', dest= 'top_k', type=int, default= 1)
parser.add_argument('--category_names', action='store', dest= 'category_names', default= 'cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default= False)

args = parser.parse_args()

image_path= args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
device = 'cuda' if args.gpu == True else 'cpu'

print('Processing the image..')
tensor_image = utils.process_image(image_path)

print('Loading the model..')
model = utils.load_model(checkpoint, device)

print('Results')
probs, classes = utils.predict(image_path, model, top_k, device)
utils.output_predictions(probs, classes, category_names)
print('Prediction completed!')