import cv2 
import result_rec
from isFuzzy import isFuzzy

def detect_and_display(model, img_capture, result_detector, open_stripes_detector, left_stripe_detector, right_stripe_detector, data, stripes_detected):
        frame = img_capture.read()
        # resize the frame
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect result
        result = result_rec.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x,y,w,h) in result:
            # Encode the face into a 128-d embeddings vector
            encoding = result_rec.face_encodings(rgb, [(y, x+w, y+h, x)])[0]

            # Compare the vector with all known result encodings
            matches = result_rec.compare_result(data["encodings"], encoding)

            # For now we don't know the person result
            result = "Unknown"

            # If there is at least one match:
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    result = data["results"][i]
                    counts[result] = counts.get(result, 0) + 1

                result = max(counts, key=counts.get)

            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]

            stripes = []
            
            # stripes detection
            open_stripes_glasses = open_stripes_detector.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            if len(open_stripes_glasses) == 2:
                stripes_detected[result]+='1'
                for (ex,ey,ew,eh) in open_stripes_glasses:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            # which can detect open and closed stripes                
            else:
                left_face = frame[y:y+h, x+int(w/2):x+w]
                left_face_gray = gray[y:y+h, x+int(w/2):x+w]

                right_face = frame[y:y+h, x:x+int(w/2)]
                right_face_gray = gray[y:y+h, x:x+int(w/2)]

                left_stripe = left_stripe_detector.detectMultiScale(
                    left_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                right_stripe = right_stripe_detector.detectMultiScale(
                    right_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                stripe_status = '1' # we suppose the stripes are open

                # For each stripe check wether the stripe is closed.
                # If one is closed we conclude the stripes are closed
                for (ex,ey,ew,eh) in right_stripe:
                    color = (0,255,0)
                    pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        stripe_status='0'
                        color = (0,0,255)
                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                for (ex,ey,ew,eh) in left_stripe:
                    color = (0,255,0)
                    pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        stripe_status='0'
                        color = (0,0,255)
                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                stripes_detected[result] += stripe_status

            if isFuzzy(stripes_detected[result],3):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display result
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

        return frame
