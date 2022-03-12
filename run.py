import os, argparse
from train import train_run
from test import test_run



def mode_type(val):

	if val not in ['train','test','help']:

		msg = "Mode should either be [train,test,pred,help]. Run ** python run.py -mode help ** for help"

		raise argparse.ArgumentTypeError(msg)
	return val

def data_type(val):

	if val not in ['captions','concepts','concepts_seq']:

		msg = "Datatype should either be [captions, concepts_seq, concepts]. Run ** python run.py -mode help ** for help"

		raise argparse.ArgumentTypeError(msg)
	return val

def imagepath_findable(path):

	if not os.path.exists(path):
		msg = "image path does not exist"

		raise argparse.ArgumentTypeError(msg)
	return path

def model_type(val):

	if val not in ['transformer','concept_classifier']:
		msg = "Model should either be [transformer,concept_classifier]. Run ** python run.py -mode help ** for help"

		raise argparse.ArgumentTypeError(msg)
	return val

if __name__ == "__main__":

	run = False

	parser = argparse.ArgumentParser(description="--run_args")


	parser.add_argument('-mode',type=mode_type)
	parser.add_argument('--traincsv')
	parser.add_argument('--imagepath',type=imagepath_findable)
	parser.add_argument('--batchsize',default=1)
	parser.add_argument('--epochs',default=10)
	parser.add_argument('--model',type=model_type)
	parser.add_argument('--modeloutput',default='CLEF_MODEL')
	parser.add_argument('--valsplit',default=0.25)
	parser.add_argument('--gt_csv')
	parser.add_argument('--classweight_hbar', default=1000)
	parser.add_argument('--classweight_lbar', default = 0.05)
	parser.add_argument('--data_type', type=data_type)

	args = parser.parse_args()




	if args.mode == "test":
		run = True
		test_run()
	elif args.mode == "train":
		run = True
		train_filename = args.traincsv
		if train_filename is not None:
			if args.data_type == "concepts" and args.model == "transformer":
				raise("Concept transformer sequence generation only works with --data_type [concepts_seq]")
			train_run(train_filename, 
						args.imagepath,
						int(args.batchsize),
						int(args.epochs), 
						args.valsplit, 
						[args.classweight_hbar,args.classweight_lbar],
						args.model,
						args.modeloutput,
						args.data_type)
		else:
			msg = "Must include --traincsv. run ** python run.py -mode help ** for help"
			raise argparse.ArgumentTypeError(msg)

	elif args.mode == "help":
		run = True
		f = open('README.md', 'r')
		help_ = f.read()
		print(help_)
	if not run:
		print("\nrun ** python run.py -mode help ** for help")
	