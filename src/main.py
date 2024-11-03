import tkinter as tk
from gui.app import ISPApplication

def main():
    root = tk.Tk()
    app = ISPApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()