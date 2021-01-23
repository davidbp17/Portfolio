# Overview

This portfolio consists of some of the projects I have been working on in the past couple years through my graduate education. These projects deal with many important sectors of Computer Science such as Speech Processing, Big Data & Analysis, Design Patterns and Machine Learning. There are some projects that was greatly influenced from previous jobs where I did GUI development for devices. I believe that it is quite important that users have a understandable interface and be able to visualize important data, and incorporate that into my work. 


## Board Scripting Language
### Scripting language built for communication between a GUI and HID Device

This project started as a suggestion from my boss about what we wanted to include into a C# GUI that I was designing. It was decided against after a few days of planning due to the project being time consuming and not a necessary component. Given that this idea was trying to be implemented in previous jobs, I began to work on this on my own time. My general idea was based off the Arduino coding system, a C style language that gives the ability to read and write to a device, delcare variables, implement delays, and loop. This gives the engineers that are using the GUI a familiar coding style.
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
### Intermediary Code
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
