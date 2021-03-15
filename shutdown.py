import RPi.GPIO as GPIO
import os
import time

P1 = 13
P2 = 11

def main():
    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(P1, GPIO.OUT)
        GPIO.setup(P2, GPIO.IN)

        count = 1

        while True:
            if GPIO.input(P2):
                print(count)
                count += 1
            else:
                GPIO.output(P1, GPIO.HIGH)
                print('SHUTTING DOWN')
                time.sleep(5)
                shutdown()
                break
            
    except Exception as e:
        print(e)
    
    finally:
        GPIO.cleanup()

def shutdown():
    os.system(r'sudo shutdown -h now')

if __name__ == '__main__':
    main()