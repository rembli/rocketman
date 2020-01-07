import pygame
from soundshift import pitch
import writelabels as wl
from scipy.io import wavfile
import os
import shutil

# DATA
data_root = "C:\\Data\\Dev-Data\\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")
files_path = data_root + "cluster\\"
files_path_target = data_root + "labels\\"

# DISPLAY SETTINGS
screen_caption = "Mechanical Turk"
screen_bg_color = (255, 255, 255)
screen_width = 400
screen_height = 512


# MAIN
def main ():

    print ('''
        HOW-TO
        ==========================================================
        Up:     Move note up
        Down:   Move note down
        ----------------------------------------------------------
        Right:  Next note
        Left:   Previous note
        ----------------------------------------------------------
        Space:  Select note
        x:      Select that this pic is not a note
        y:      Select that this pic should be excluded from training
        ==========================================================
        ESC:    Quit mechanical turk
    ''')


    # INIT SOUNDMIXER (important: before paygame.init!)
    sampleRate = 44100
    pygame.mixer.pre_init(sampleRate, -16, 1)

    # INIT SCREEN
    pygame.init()
    pygame.display.set_caption (screen_caption)
    screen = pygame.display.set_mode ((screen_width,screen_height))

    # LOAD NOTE
    note_positions = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
    note_names = ["a", "g", "f", "e", "d", "c", "h", "a", "g", "f", "e", "d", "c", "h", "a", "g", "f", "e", "d", "c", "h", "a", "g", "f", "e", "d", "c", "h"]
    num_notes = len (note_positions)
    current_pos = 12
    is_selected = False
    is_note = True

    fps, sound = wavfile.read(data_root+"sounds/bowl.wav")
    transposed_sounds = [pitch(sound, n) for n in range (num_notes, 0, -1)]
    sounds = list (map(pygame.sndarray.make_sound, transposed_sounds))

    # LOAD FILES
    filenames = os.listdir(files_path)
    current_image = 0
    key = ' '

    while True:
        #######################################################################
        # DRAW SCREEN
        #######################################################################

        screen.fill(screen_bg_color)

        ## rescale and draw image of note
        image = pygame.image.load(files_path + filenames[current_image])
        image_width, image_height = image.get_rect().size

        image_scaling_factor = (screen_height * 0.8) / image_height
        image = pygame.transform.scale(image, (int (image_width * image_scaling_factor), int (image_height * image_scaling_factor)))
        image_width, _ = image.get_rect().size

        screen.blit(image, (10, 10))

        ## draw seperator line
        line_color = (0 ,0, 0)
        pygame.draw.line (screen, line_color, (screen_width/2, 0), (screen_width/2, screen_height))

        ## draw lines for notes
        w1 = int(screen_width/2 + 20)
        w2 = screen_width - 20
        pygame.draw.line(screen, line_color, (w1, 40), (w2, 40))
        pygame.draw.line(screen, line_color, (w1, 60), (w2, 60))
        pygame.draw.line(screen, line_color, (w1, 80), (w2, 80))
        pygame.draw.line(screen, line_color, (w1, 100), (w2, 100))
        pygame.draw.line(screen, line_color, (w1, 120), (w2, 120))

        pygame.draw.line(screen, line_color, (w1, 300), (w2, 300))
        pygame.draw.line(screen, line_color, (w1, 320), (w2, 320))
        pygame.draw.line(screen, line_color, (w1, 340), (w2, 340))
        pygame.draw.line(screen, line_color, (w1, 360), (w2, 360))
        pygame.draw.line(screen, line_color, (w1, 380), (w2, 380))

        ## print note name
        if is_note:
            font = pygame.font.SysFont("comicsansms", 72)
            font_color = (0, 128, 0)
            text = font.render(note_names[current_pos], True, font_color)
            screen.blit(text, (w1 + 60, 160))
        else:
            font = pygame.font.SysFont("comicsansms", 45)
            font_color = (255, 0, 0)
            text = font.render("No note", True, font_color)
            screen.blit(text, (w1, 160))

        ## draw note
        if is_selected:
            pygame.draw.circle(screen, (0, 128, 0), (w1 + 85, note_positions[current_pos]), 10)
        else:
            pygame.draw.circle(screen, (255, 0, 0), (w1 + 85, note_positions[current_pos]), 10)

        ## updated screen
        pygame.display.flip()

        #######################################################################
        # DO ACTIONS
        #######################################################################

        if is_selected or not is_note:
            filepath = files_path + filenames[current_image]

            if is_selected:
                if not os.path.isdir(files_path_target + str (current_pos)+ "\\"):
                    os.mkdir(files_path_target + str (current_pos)+ "\\")
                filepath_target = files_path_target + str (current_pos)+ "\\" + filenames[current_image]
            else:
                if key == "x":
                    filepath_target = files_path_target + "other\\" + filenames[current_image]
                else:
                    filepath_target = files_path_target + "excluded\\" + filenames[current_image]

            print("copy",filepath,"to",filepath_target)
            shutil.copy (filepath, filepath_target)

            current_image = current_image + 1
            if current_image >= len(filenames):
                current_image = 0

            is_note = True
            is_selected = False

        #######################################################################
        # LISTEN TO EVENTS
        #######################################################################

        event = pygame.event.wait()

        if event.type == pygame.QUIT:
            wl.write_labels(files_path_target)
            pygame.quit()

        if event.type == pygame.KEYDOWN:
            key = pygame.key.name(event.key)
            # print (key)

            if event.key == pygame.K_ESCAPE:
                wl.write_labels (files_path_target)
                pygame.quit()
                raise KeyboardInterrupt

            if event.key == pygame.K_RIGHT:
                current_image = current_image + 1
                if current_image >= len(filenames):
                    current_image = 0

            if event.key == pygame.K_LEFT and current_image > 0:
                current_image = current_image - 1

            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                if event.key == pygame.K_UP:
                    current_pos = current_pos - 1
                    if current_pos < 0:
                        current_pos = len(note_positions) - 1

                if event.key == pygame.K_DOWN:
                    current_pos = current_pos + 1
                    if current_pos >= len(note_positions):
                        current_pos = 0

                # sound.play(fade_ms=25)
                # sounds[note_current_pos].play(fade_ms=50)

            if key == "x" or key == "y" or key == "z":
                if is_note:
                    is_note = False
                else:
                    is_note = True

            if event.key == pygame.K_SPACE:
                if is_selected:
                    current_image = current_image + 1
                else:
                    is_selected = True
                    sounds[current_pos].play(fade_ms=50)


if __name__ == '__main__':
    try:
        print ("startup ...")
        main()
    except KeyboardInterrupt:
        print('.... shut down.')

