# Overview

This portfolio consists of some of the projects I have been working on in the past couple years through my graduate education. These projects deal with many important sectors of Computer Science such as Speech Processing, Big Data & Analysis, Design Patterns and Machine Learning. There are some projects that was greatly influenced from previous jobs where I did GUI development for devices. I believe that it is quite important that users have a understandable interface and be able to visualize important data, and incorporate that into my work. 


## Board Scripting Language
### Scripting language built for communication between a GUI and HID Device

This project started as a suggestion from my boss about what we wanted to include into a C# GUI that I was designing. It was decided against after a few days of planning due to the project being time consuming and not a necessary component. Given that this idea was trying to be implemented in previous jobs, I began to work on this on my own time. This project is one of my favorites because it touched upon many aspects of undergraduate and graduate education such as Computability Theory, Data Structurs, Algorithms, Coding Structure, Grammars, and Design Patterns. My general idea for the scripting language was based off the Arduino coding system, a C style language that gives the ability to read and write to a device, delcare variables, implement delays, and loop. This gives the engineers that are using the GUI a familiar coding style.

There are many aspects of the language built to be useful to intended users. The scripting language would be an interpretive language internally, but all code would be run through once to effectively compile it. It is important that this scripting langauge not have errors while code is being run, as it would create a situation where restoring device values is difficult. The process of compiling and running a program consists of checking the program is closed, followed by lexical analysis and converting to intermediate code, the intermediate code is checked once for any syntax errors, then the code is free to be run. There are four primitivie data types which are int, double, string and bool and each has a corresponding array type. Keeping with C like languages there is implicit type conversion and type promotion. I also felt that it was important to not implement mandatory spacing in for, while and if else blocks.

### Sample Code

```c
int readVal = 0;
bool loopControl = true;
if(loopControl){
    for(int reg = 0x1000; reg <= 0x100F; reg++)
    {
        readVal = Read(reg);
        PrintLine("Register Number: " + reg);
        PrintLine("Register Value: " + readVal);
    }
}
else
    PrintLine(Read(0x1000));

```
### Intermediate Code
*Note the line numbers in the intermediary code start at line 0*
```c
int readVal = 0;
bool loopControl = true;
if(loopControl) else goto 10;
int reg = 4096;
if(reg <= 4011) else goto 10;
readVal = Read(reg);
PrintLine("Register Number: " + reg);
PrintLine("Register Value: " + readVal);
reg = reg + 1;
goto 4;
goto 12;
PrintLine(Read(4096));

```

The format of the intermediate code allows for easy and consistent parsing as all lines are now assignments, function calls, branch if statements, or break/continue statements. Internally each line of the intermediate code keeps track of the original line number in the program it belonged to, this allows errors to be tracked correctly. From here the class proceeds to parse and evaluate expressions as needed. Evaluation of expressions requires that order of operations be preserved, so the expression is converted into Polish notation before being evaluated.

There are few elements of the scripting language that are incomplete, such as user defined methods haven't been implemented yet. There are also elements of languages that are left out, such as refrences, structs, const variables, variables with different scopes, declarations as well as header files. The reason is that those elements of languages aren't necessary for what this scripting language does, and would take a lot of extra time to implement. Any potential user who needed that level of complexity would likely code it themselves using another language or ask to have it manually programed in the interface. Another gripe I have about the implementation of the Board Scripting Language would be how I tokenized the lines, I struggled using Regex to split strings that contained the program lines. There are parts in code that I wasn't optimally splitting lines. Lines were getting split by certain chars as opposed to substrings, and this creates a lot of empty strings, extra code to account for the splits, and strings elements that need to be stitched together which made the original lexical analysis more complex than was necessary. If I were to redo this project, the biggest element I would work on is how the tokenization is done, or even create my own version. Last of all, comments aren't internally processed by the Board Scripting Language, but its the job of the GUI to not pass in comments into the program.

Overall the code works very well, when attached to C# GUI using WPF as I did for testing this out. I left the GUI code and HID code out of this project on Github purposefully because it wouldn't make sense to test it unless you had a compatable device with correct firmware, as well as greatly increases the amount of code and complexity. There are two extra classes, one is the abstract Bridge pattern that I use to communicate with the output window in the GUI, that is used so that the scripting language doesnt have access to the entire MainWindow and all elements, and the second is the Math Operations class, which has static methods for composing arithmatic, string and boolean operations. Overall I am confident this scripting language works very well. Below is the link for the library

[Link to Board Scripting Language Library](./BoardScriptingLanguage).
