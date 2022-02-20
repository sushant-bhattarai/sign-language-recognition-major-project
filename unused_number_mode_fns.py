def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	operator = ""
	if pred_text == 1:
		operator = "+"
	elif pred_text == 2:
		operator = "-"
	elif pred_text == 3:
		operator = "*"
	elif pred_text == 4:
		operator = "/"
	elif pred_text == 5:
		operator = "%"
	elif pred_text == 6:
		operator = "**"
	elif pred_text == 7:
		operator = ">>"
	elif pred_text == 8:
		operator = "<<"
	elif pred_text == 9:
		operator = "&"
	elif pred_text == 0:
		operator = "|"
	return operator


def calculator_mode(cam):
	global is_voice_on
	flag = {"first": False, "operator": False, "second": False, "clear": False}
	count_same_frames = 0
	first, operator, second = "", "", ""
	pred_text = ""
	calc_text = ""
	info = "Enter first number"
	Thread(target=say_text, args=(info,)).start()
	count_clear_frames = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_pred_text = pred_text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				pred_text = get_pred_from_contour(contour, thresh)
				if old_pred_text == pred_text:
					count_same_frames += 1
				else:
					count_same_frames = 0

				if pred_text == "C":
					if count_same_frames > 5:
						count_same_frames = 0
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"
						Thread(target=say_text, args=(info,)).start()

				elif pred_text == "Best of Luck " and count_same_frames > 15:
					count_same_frames = 0
					if flag['clear']:
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"
						Thread(target=say_text, args=(info,)).start()
					elif second != '':
						flag['second'] = True
						info = "Clear screen"
						#Thread(target=say_text, args=(info,)).start()
						second = ''
						flag['clear'] = True
						try:
							calc_text += "= "+str(eval(calc_text))
						except:
							calc_text = "Invalid operation"
						if is_voice_on:
							speech = calc_text
							speech = speech.replace('-', ' minus ')
							speech = speech.replace('/', ' divided by ')
							speech = speech.replace('**', ' raised to the power ')
							speech = speech.replace('*', ' multiplied by ')
							speech = speech.replace('%', ' mod ')
							speech = speech.replace('>>', ' bitwise right shift ')
							speech = speech.replace('<<', ' bitwise leftt shift ')
							speech = speech.replace('&', ' bitwise and ')
							speech = speech.replace('|', ' bitwise or ')
							Thread(target=say_text, args=(speech,)).start()
					elif first != '':
						flag['first'] = True
						info = "Enter operator"
						Thread(target=say_text, args=(info,)).start()
						first = ''

				elif pred_text != "Best of Luck " and pred_text.isnumeric():
					if flag['first'] == False:
						if count_same_frames > 15:
							count_same_frames = 0
							Thread(target=say_text, args=(pred_text,)).start()
							first += pred_text
							calc_text += pred_text
					elif flag['operator'] == False:
						operator = get_operator(pred_text)
						if count_same_frames > 15:
							count_same_frames = 0
							flag['operator'] = True
							calc_text += operator
							info = "Enter second number"
							Thread(target=say_text, args=(info,)).start()
							operator = ''
					elif flag['second'] == False:
						if count_same_frames > 15:
							Thread(target=say_text, args=(pred_text,)).start()
							second += pred_text
							calc_text += pred_text
							count_same_frames = 0

		if count_clear_frames == 30:
			first, second, operator, pred_text, calc_text = '', '', '', '', ''
			flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
			info = "Enter first number"
			Thread(target=say_text, args=(info,)).start()
			count_clear_frames = 0

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, "Predicted text- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
		cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
		if is_voice_on:
			cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		else:
			cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('t'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('t'):
		return 1
	else:
		return 0


