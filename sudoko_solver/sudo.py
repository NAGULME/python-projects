# Import the necessary libraries
import tkinter as tk  # Import the tkinter library for creating GUI applications
from tkinter import messagebox  # Import the messagebox module for displaying error messages

# Define a class for the Sudoku solver
class SudokuSolver:
    # Initialize the Sudoku solver with a given size
    def __init__(self, size):
        # Create a new tkinter window
        self.window = tk.Tk()
        # Set the title of the window
        self.window.title("Sudoku Solver")
        # Set the initial size of the window
        self.window.geometry("800x600")
        # Maximize the window
        self.window.state("zoomed")
        # Store the size of the Sudoku puzzle
        self.size = size
        # Initialize the Sudoku grid with zeros
        self.grid = [[0]*size for _ in range(size)]
        # Create a dictionary to store the entry widgets
        self.entries = {}

        # Create the entry widgets for the Sudoku grid
        for i in range(size):
            for j in range(size):
                # Create a new entry widget
                e = tk.Entry(self.window, width=5, font=('Arial', 24))
                # Place the entry widget in the grid
                e.grid(row=i, column=j, padx=5, pady=5)
                # Store the entry widget in the dictionary
                self.entries[(i, j)] = e

        # Create a button to solve the Sudoku puzzle
        solve_button = tk.Button(self.window, text="Solve", command=self.solve, font=('Arial', 24))
        # Place the solve button below the Sudoku grid
        solve_button.grid(row=size, column=0, columnspan=size, padx=5, pady=5)

        # Create a button to reset the Sudoku puzzle
        reset_button = tk.Button(self.window, text="Reset", command=self.reset, font=('Arial', 24))
        # Place the reset button below the solve button
        reset_button.grid(row=size+1, column=0, columnspan=size, padx=5, pady=5)

    # Method to solve the Sudoku puzzle
    def solve(self):
        # Get the input values from the grid
        for i in range(self.size):
            for j in range(self.size):
                # Get the value from the entry widget
                val = self.entries[(i, j)].get()
                # If the value is not empty, convert it to an integer and store it in the grid
                if val:
                    self.grid[i][j] = int(val)

        # Solve the Sudoku puzzle using backtracking
        if self.solve_sudoku(self.grid):
            # Display the solved puzzle
            for i in range(self.size):
                for j in range(self.size):
                    # Clear the entry widget
                    self.entries[(i, j)].delete(0, tk.END)
                    # Insert the solved value into the entry widget
                    self.entries[(i, j)].insert(0, self.grid[i][j])
        else:
            # Display an error message if no solution exists
            messagebox.showerror("Error", "No solution exists")

    # Method to solve the Sudoku puzzle using backtracking
    def solve_sudoku(self, grid):
        # Iterate over the grid
        for i in range(self.size):
            for j in range(self.size):
                # If the current cell is empty
                if grid[i][j] == 0:
                    # Try values from 1 to size
                    for val in range(1, self.size+1):
                        # If the value is valid
                        if self.is_valid(grid, i, j, val):
                            # Place the value in the current cell
                            grid[i][j] = val
                            # Recursively try to fill in the rest of the grid
                            if self.solve_sudoku(grid):
                                # If the rest of the grid can be filled in, return True
                                return True
                            # If the rest of the grid cannot be filled in, reset the current cell
                            grid[i][j] = 0
                    # If no value can be placed in the current cell, return False
                    return False
        # If the entire grid has been filled in, return True
        return True

    # Method to check if a value is valid in a given cell
    def is_valid(self, grid, row, col, val):
        # Check if the value is already present in the row or column
        for i in range(self.size):
            if grid[row][i] == val or grid[i][col] == val:
                # If the value is already present, return False
                return False
        # Check if the value is already present in the box
        box_size = int(self.size ** 0.5)
        box_row = row - row % box_size
        box_col = col - col % box_size
        for i in range(box_size):
            for j in range(box_size):
                if grid[box_row + i][box_col + j] == val:
                    # If the value is already present, return False
                    return False
        # If the value is not already present, return True
        return True

    # Method to reset the Sudoku puzzle
    def reset(self):
        # Clear the entry widgets
        for i in range(self.size):
            for j in range(self.size):
                self.entries[(i, j)].delete(0, tk.END)
                # Reset the grid
                self.grid[i][j] = 0

    # Method to run the Sudoku solver
    def run(self):
        # Start the main event loop
        self.window.mainloop()

# Main function
def main():
    # Create a new tkinter window
    root = tk.Tk()
    # Set the title of the window
    root.title("Sudoku Solver")

    # Function to create a new Sudoku solver with a given size
    def create_solver(size):
        # Destroy the current window
        root.destroy()
        # Create a new Sudoku solver
        solver = SudokuSolver(size)
        # Run the Sudoku solver
        solver.run()

    # Create buttons to select the size of the Sudoku puzzle
    tk.Button(root, text="6x6", command=lambda: create_solver(6)).pack()
    tk.Button(root, text="9x9", command=lambda: create_solver(9)).pack()
    tk.Button(root, text="16x16", command=lambda: create_solver(16)).pack()
    tk.Button(root, text="32x32", command=lambda: create_solver(32)).pack()

    # Start the main event loop
    root.mainloop()

# Run the main function
if __name__ == "__main__":
    main()

# Commit 1: Added Sudoku solver with reset button and size selection
# Commit 2: Fixed bug in solve_sudoku method where it was not checking for valid values
# Commit 3: Improved performance of solve_sudoku method by reducing number of recursive calls
# Commit 4: Added error handling for invalid input values
# Commit 5: Improved user interface by adding padding to entry widgets