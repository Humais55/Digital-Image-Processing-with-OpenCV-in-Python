import cv2

#Read an image from disk 
image_path='./img.png'
original_image = cv2.imread(image_path)

# Check if the image was successfully loaded 
if original_image is None:
    print("Error: Unable to load the image.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection using the Canny edge detector 
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Display the original and processed images 
    cv2.imshow('Original Image', original_image) 
    cv2.imshow('Grayscale Image', gray_image) 
    cv2.imshow('Edge Detection', edges)
    
    # wait for a key press and close the OpenCV windows 
    cv2.waitKey(0)
    cv2.destroyAllWindows()