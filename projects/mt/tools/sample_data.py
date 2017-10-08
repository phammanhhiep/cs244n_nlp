def sample_data (source_name, target_name, source_sample_name, target_sample_name, size=2000):
	source = []
	count = 0
	with open (source_name) as s, open (target_name) as t, open (source_sample_name, 'w') as ss, open (target_sample_name, 'w') as st:
		for si,ti in zip (s, t):
			count += 1
			ss.write (si)
			st.write (ti)
			if count == size:
				break


if __name__ == '__main__':
	target = 'data/europarl.fr-en.en/europarl.fr-en.en'
	source = 'data/europarl.fr-en.fr/europarl.fr-en.fr'
	target_sample = 'data/en-fr-sample/en'
	source_sample = 'data/en-fr-sample/fr'
	sample_data (source, target, source_sample, target_sample, 200)


