import pickle
import argparse
import numpy as np
import random
import pandas as pd
# import nlpaug.augmenter.word as naw
from transformers import AutoTokenizer
import os
from sklearn.model_selection import train_test_split
import py_vncorenlp

# def build_vncorenpl(save_dir):
#     if not os.path.exists(save_dir):
#         ValueError("Save dir does not exist!")
#     py_vncorenlp.download_model(save_dir=r'/data2/npl/ICEK/VnCoreNLP') 
#     model = py_vncorenlp.VnCoreNLP(save_dir=r'D:\Code\Python\VnCoreNLP')
#     return model

# def word_segmentate(model, text)->str:
#     return model.word_segment(text)[0]


def preprocess_data(dataset,tokenizer_type,w_aug,aug_type):
	os.makedirs("preprocessed_data", exist_ok=True)
	if dataset == "ViIHSD":
		class3int = {'Non HS': 0 ,'Implicit HS': 1, 'Explicit HS':2}

		data_dict = {}
		data_home = "./data/"

		for datatype in ["train","val","test"]:

			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile)

			label,text = [],[]
			aug_sent1_of_post = []

			for i,one_class in enumerate(data["label"]):
				# print("i: ", i)
				# print("one_class: ", one_class)
				
				label.append(class3int[one_class])
				text.append(data["text"][i])
				
			if datatype == "train" and w_aug:
				for i, one_aug_sent in enumerate(data["aug_sent1_of_post"]):
					aug_sent1_of_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(text).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(aug_sent1_of_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,aug_sent1_of_post)]
				combined_label = [list(i) for i in zip(label,label)]

				processed_data = {}

				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_text =tokenizer.batch_encode_plus(text).input_ids

				processed_data = {}

				processed_data["tokenized_text"] = tokenized_text
				processed_data["label"] = label
				processed_data["text"] = text

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/ihc_pure_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
			f.close()
		else:
			with open("./preprocessed_data/ViIHSD_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
				f.close()

if __name__ == '__main__':
	# vncorenpl_model= build_vncorenpl(r'/data2/npl/ICEK/VnCoreNLP')
	parser = argparse.ArgumentParser(description='Enter tokenizer type')

	parser.add_argument('-d', default="ViIHSD",type=str,
				   help='Enter dataset')

	parser.add_argument('-t', default="/data2/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/phobert-base",type=str,
# 	parser.add_argument('-t', default="roberta-base",type=str,
				   help='Enter tokenizer type')

	parser.add_argument('--aug_type', default="syn",type=str,
				   help='Enter augmentation type')
	parser.add_argument('--aug', action='store_true')
	args = parser.parse_args()

	preprocess_data(args.d,args.t,w_aug=args.aug,aug_type=args.aug_type)


# from py_vncorenlp import VnCoreNLP
# import pandas as pd
# import pickle
# import argparse
# import numpy as np
# import random
# # import nlpaug.augmenter.word as naw
# # from transformers import AutoTokenizer
# import random
# import os
# from sklearn.model_selection import train_test_split
# import py_vncorenlp

# # Automatically download VnCoreNLP components from the original repository
# # and save them in some local working folder
# py_vncorenlp.download_model(save_dir=r'D:\Code\Python\VnCoreNLP')

# # Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models` 
# model = py_vncorenlp.VnCoreNLP(save_dir=r'D:\Code\Python\VnCoreNLP')

# # vncorenlp_path = r"D:\Code\Python\VnCoreNLP"  # Thay đổi đường dẫn tới thư mục chứa VnCoreNLP
# # annotator = VnCoreNLP(vncorenlp_path)

# # Văn bản cần phân tách
# text = "Hôm nay là một ngày đẹp trời."

# # Sử dụng VnCoreNLP để phân tách từ
# word_seg = model.word_segment(text)

# # In ra kết quả
# print("Word Segmentation: ", word_seg)
