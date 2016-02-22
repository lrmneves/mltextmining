import os

with open("final_result.txt","a") as final_result:
	for root, dirs, files in os.walk("."):
		for f in files:
			if not f.endswith(".results.txt"):
				continue
			query = f.split(".")[0]
			with open(os.path.join(root, f)) as current_file:
				for line in current_file.readlines():
					doc = line.split(" ")
					final_result.write(query + " " + " ".join(doc[1:]))
