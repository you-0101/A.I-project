#############################
univ_id = "0000"           ##
#############################

from tkinter import Tk, Canvas, Frame, Button, BOTH, TOP, LEFT, RIGHT, BOTTOM
from Problem import Problem
import random
import sudoku_50 as problem_set

MARGIN = 20  # Pixels around the board
SIDE = 50  # Width of every board cell.
WIDTH = HEIGHT = MARGIN * 2 + SIDE * 9  # Width and height of the whole board
DELAY = 0.05  # the delay time makes the changing number visible
#PROBLEM_NUM = problem_set.problem_num

class SudokuUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.__initUI()

    def __initUI(self):
        self.parent.title("AI: Sudoku Solver")
        self.pack(fill=BOTH)
        self.canvas = Canvas(self, width=WIDTH, height=HEIGHT + 10)
        self.easy_problem = Problem(self.canvas, 0, DELAY, random.randint(0, 1), setting = "easy")
        self.hard_problem = Problem(self.canvas, 0, DELAY, random.randint(0, 1), setting = "hard")

        self.problem = None # Set this by click button
        
        
        # Initialize each cell in the puzzle
        for i in range(1, 10):
            for j in range(1, 10):
                self.item = self.canvas.create_text(
                    MARGIN + (j - 1) * SIDE + SIDE / 2, MARGIN + (i - 1) * SIDE + SIDE / 2,
                    text='', tags="numbers", fill="black", font=("Helvetica", 12)
                )
        self.item = self.canvas.create_text(40, 490, text='Count :', fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(95, 490, text='', fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(170, 490, text='Average :', fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(225, 490, text='',fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(320, 490, text='Ranking :',fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(370, 490, text='', fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(420, 490, text='Total :', fill="black", font=("Helvetica", 13))
        self.item = self.canvas.create_text(460, 490, text='', fill="black", font=("Helvetica", 13))
        self.canvas.pack(fill=BOTH, side=TOP)
        self.start_button1 = Button(self, text="__Hard__", command=self.__start_hard_solver)
        self.start_button2 = Button(self, text="__Easy__", command=self.__start_easy_solver)
        
        self.start_button2.pack(side=LEFT)
        self.start_button1.pack(side=RIGHT)
        #self.start_button2.config(state="disabled")
        self.__draw_grid()

    # Draws 9x9 grid
    def __draw_grid(self):
        for i in range(10):
            width = 3 if i % 3 == 0 else 1
            x0 = MARGIN + i * SIDE
            y0 = MARGIN
            x1 = MARGIN + i * SIDE
            y1 = HEIGHT - MARGIN
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=width)
            x0 = MARGIN
            y0 = MARGIN + i * SIDE
            x1 = WIDTH - MARGIN
            y1 = MARGIN + i * SIDE
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=width)
    def __start_easy_solver(self):
      self.problem = self.easy_problem
      self.__start_solver()
      
    def __start_hard_solver(self):
      self.problem = self.hard_problem
      self.__start_solver()
      
      
    def __start_solver(self):
        self.start_button1.config(state="disabled")
        self.start_button2.config(state="disabled")
        for i in range(self.problem.problem_num):
            for m in range(1, 10):
                for n in range(1, 10):
                    if(self.problem.given_number[m-1][n-1] != 0): self.canvas.itemconfig(9 * (m - 1) + n, text=self.problem.given_number[m-1][n-1], tags="numbers", fill="blue")
                    else:self.canvas.itemconfig(9 * (m - 1) + n, text='', tags="numbers", fill="black")
                    
            self.solver_class = solver_class(self.problem)
            self.solver_class.solver()
            if self.problem.finished==0:
                self.problem.fail()
                return
            self.canvas.update()
            
            
            if(i != self.problem.problem_num -1):  self.problem = Problem(self.canvas, self.problem.tk, 0.0, self.problem.temp, setting = self.problem.setting)
        self.problem.update_a()
        #self.start_button2.config(state="active")
        
        #If the problem has finished, this function will display "Finished!"
        self.problem.is_done()
        
    def __submit(self):
        request=self.problem.submit(univ_id, "")
        message=request.split(',')
        if int(message[0]) == 100:
            self.problem.fail_10min()
        elif int(message[0]) == 101:
            self.canvas.update()
            self.canvas.itemconfig(87, text=int(message[1]), tags="numbers", fill="blue")
            self.canvas.itemconfig(89, text=int(message[2]), tags="numbers", fill="blue")
            self.problem.already_done()
        elif int(message[0]) == 102:
            self.canvas.update()
            self.canvas.itemconfig(87, text=int(message[1]), tags="numbers", fill="blue")
            self.canvas.itemconfig(89, text=int(message[2]), tags="numbers", fill="blue")
            self.problem.is_done()
        elif int(message[0]) == 501:
            #   print "501"
            self.problem.wrong_id_pw()

class solver_class():
    def __init__(self, problem):
        self.problem = problem
        self.given_number = problem.given_number

    def solver(self):
        self.puzzle = []
        new = []
        for i in range(0, 9):
            for j in range(0, 9):
                if(self.given_number[i][j] != 0): new.append(self.given_number[i][j])
                else: new.append(0)
            self.puzzle.append(new)
            new = []


        # TO DO: need to write solver
        # Your code goes here...
        # this is example using for loop
        # you have to improve this code

        self.row_check = [[0]*10 for i in range(9)]     #setting for check row
        self.col_check = [[0]*10 for i in range(9)]     #setting for check col
        self.square_check = [[0]*10 for i in range(9)]  #setting for check square
        self.square_num =[[0,0,0,1,1,1,2,2,2],
                     [0,0,0,1,1,1,2,2,2],
                     [0,0,0,1,1,1,2,2,2],
                     [3,3,3,4,4,4,5,5,5],
                     [3,3,3,4,4,4,5,5,5],
                     [3,3,3,4,4,4,5,5,5],
                     [6,6,6,7,7,7,8,8,8],
                     [6,6,6,7,7,7,8,8,8],
                     [6,6,6,7,7,7,8,8,8]]


                     
        self.zero_xy = []
        zero_cnt = 0
        output=0
            
        '''#Checking the first given number and if there is given number putting each part of
          row, col, square. Then we will not use that number to each part when we find out the number.
          Further more we will make array of blank pos. Each array will have information of xpos and ypos'''
        for r in range(1,10):
            for c in range(1,10):
                if self.given_number[r-1][c-1] ==0: # If it is blank!!
                    self.zero_xy.append((r-1,c-1))  # Put into array of xpos, ypos
                    zero_cnt +=1                    # Increase number of blank    
    
                else:
                    self.row_check[r-1][self.puzzle[r-1][c-1]]=1                            # Will not use in row
                    self.col_check[c-1][self.puzzle[r-1][c-1]]=1                            # Will not use in col
                    self.square_check[self.square_num[r-1][c-1]][self.puzzle[r-1][c-1]]=1   # Will not use in square

        '''#Start of infinite Loop. In this loop we will check the count of possible number of each blank
            Putting the number of minimum count and check the answer. If only one number is possible, putting that number
            But there is no only one possible the try to find 2 or 3,4 ...... and check again 1.
            When we put the number, we will erase that data and decrease the number of blank.
            Number of blank becomes 0, ends the loop'''
        while True:
            cannum=0
            zi=0
            eraze=zero_cnt
            nonum=0
            for zi in range(0, zero_cnt):
                
                r,c =self.zero_xy[zi]
   
                for num in range(1,10):
                    if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                        cannum=cannum+1

                if(cannum ==1):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==2 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                            
                elif(cannum ==3 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==4 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break 
                elif(cannum ==5 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==6 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==7 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==8 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break
                elif(cannum ==9 and nonum==0):
                    nonum=1
                    for num in range(1,10):
                        if self.row_check[r][num] == 0 and self.col_check[c][num] == 0 and self.square_check[self.square_num[r][c]][num]== 0:
                            output = self.problem.checker(r+1,c+1,num)
                            if output ==1:

                                del self.zero_xy[zi]
                                zero_cnt= zero_cnt-1
                                
                                self.puzzle[r][c]=num
                                self.row_check[r][num] = self.col_check[c][num]\
                                = self.square_check[self.square_num[r][c]][num] = 1
                                break                             
                            
                else:
                    cannum=0
                if((zi+1)==zero_cnt or zi == zero_cnt):
                    zi=0
                    break
            if(eraze==zero_cnt):
                break
                

            
                        


if __name__ == "__main__":
    root = Tk()
    SudokuUI(root)
    root.geometry("%dx%d" % (WIDTH, HEIGHT + 40))
    root.mainloop()
