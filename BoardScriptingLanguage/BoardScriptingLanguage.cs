using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Shapes;
using Windows.UI.WebUI;

	class BoardScriptingLanguage
	{

		//This scripting languange avoids from the classic stack based variable system
		//Here a dictionary keeps track of all the currently declared variables, and a stack maintains a list of variables in the current routine/subroutine
		//The variables get deleted at the end of a subroutine
		//This prohibits having variables with duplicate names with different scopes
		private Dictionary<string, object> vars;
		private Stack<List<string>> varStack;


		private HID_Connection connection;
		private BSLOutputBridge outputBridge;

		private readonly string[] datatypes = new string[] { "int", "string", "double", "bool"};
		private readonly string incr = "++";
		private readonly string decr = "--";
		private readonly string ifStat = "if";
		private readonly string whileStat = "while";
		private readonly string forStat = "for";
		private List<string> methods = new List<string> { "Read", "Write", "Print","PrintLine", "Delay" };
		private List<(int, string)> intermediateCode;
		private bool Compiled { get; set; }

		/*
		 * Board Scripting Languange class takes in a user inputed program, written in C syntax
		 * The programming language is meant to communicate with an HID board and provide an interface for users to set routines in a GUI
		 * Therefore it takes in an HID Connection class and BSLOutputAdapter for GUI control
		 * The BSLOutputBridge is an abstract class that serves to bridge the MainWindow and the BoardScriptingLanguage, without giving unnessary access to all the functions
		 * It also provides flexibility in case more output methods are needed
		 * 
		 */
		public BoardScriptingLanguage(HID_Connection tempConnection,BSLOutputBridge bridge,string[] program)
		{
			if (tempConnection == null)
				throw new ArgumentNullException("HID Connection must be instantiated");
			Compiled = false;
			connection = tempConnection;
			outputBridge = bridge;
			Compile(program);
		}
		[Serializable]
		/*
		 * Custom Exception for Program Errors
		 * Allows for tracking which line the error occured and what type of Error
		 * 
		 */
		public class ProgramException : Exception
		{
			public int LineError { get; set; }
			public ProgramException() : base(){ }
			public ProgramException(string message) : base(message) { }
			public ProgramException(string message, int line) : this(message)
			{
				LineError = line;
			}
		}


		private void Compile(string[] program)
		{
			//Compilation should reset the variable stacks
			vars = new Dictionary<string, object>();
			varStack = new Stack<List<string>>();
			//This breaks the program lines into a list of tokens to parse that have a corresponding line
			List<(int, string)> tokens = TokenizeProgram(program);
			//Match all braced statements to ensure closure
			int errorLine = VerifyClosedProgram(tokens);
			if (errorLine != -1)
				outputBridge.PrintLineOutput($"Bracket/Parentheses Error at Line {errorLine}");
			else
			{
				//Further parse out line, change loops into goto based syntax
				//Any errors will end parsing and not return anything
				intermediateCode = LexicalAnalysis(tokens);
				//Here every line is run regardless of conditional flow
				RunProgram(intermediateCode, true);
				Compiled = true;
			}
		}
		public void Run()
		{
			// Before running we check for compilation
			if(Compiled && intermediateCode != null)
				RunProgram(intermediateCode);
			else
			{
				throw new Exception("Program Not Compiled");
			}
		}

		/* Runs formated lines 
		* Compile parameter determines if commands are run or not
		* The first runthrough is for sytnax checking
		* Outputs written to GUI via bridge
		* most statement will be passed to evaluate method, which returns the calculation
		*/
		public void RunProgram(List<(int, string)> program, bool compile = false)
		{
			varStack.Push(new List<string>());
			bool breakFlag = false;
			for (int i = 0; i < program.Count; i++)
			{
				(int originalLineNumber, string line) = program[i];
				line = line.TrimStart(' ');
				string[] preprocessedLine = Regex.Split(line, @"([(),{}= ])");
				string[] processedLine = StitchSplitStrings(preprocessedLine);
				//code for processing if statement
				if (processedLine[0].Equals(ifStat))
				{
					if (processedLine[1].Equals("("))
					{
						//stack keeps track of parentheses
						//var stack needs to be added for any variable declared in loop
						varStack.Push(new List<string>());
						Stack<string> stack = new Stack<string>();
						stack.Push(processedLine[1]);
						string evalString = "";
						int j;
						//j loops through statement, keeps track of when if statement ends
						// usually for statements that are one liners such as if(true) i++;

						//the post processed line will always look like
						// if(true) goto 5 else *statement*
						for (j = 2; j < processedLine.Length; j++)
						{
							//evalString is the boolean statement to evaluate
							if (processedLine[j].Equals(")"))
							{
								stack.Pop();
								if (stack.Count == 0)
									break;
								else
									evalString += processedLine[j];
							}
							else
							{
								if (processedLine[j].Equals("("))
								{
									stack.Push(processedLine[j]);
								}
								evalString += processedLine[j];
							}
						}
						bool evalBool;
						if (stack.Count == 0)
						{
							evalBool = (bool)Evaluate(evalString, originalLineNumber, compile) && !breakFlag;
							if (!evalBool)
							{
								//goto statement should be skipped if evaluated false
								processedLine = processedLine.Skip(j + 3).ToArray();
								breakFlag = false;
							}
							else
								continue;
						}
					}
				}
				//remove empty strings at beginning of line, otherwise lengths will be off
				while (processedLine[0].Equals("") || processedLine[0].Equals(" "))
				{
					processedLine = processedLine.Skip(1).ToArray();
				}
				//else blocks are evaluated normally, its kept in post parsed code for readability
				if (processedLine[0].Equals("else"))
					processedLine = processedLine.Skip(2).ToArray();
				//Code for processing general loop statements/exits
				if (processedLine[0].Equals("break") || processedLine[0].Equals("continue") || processedLine[0].Equals("goto"))
				{
					if (processedLine[0].Equals("goto"))
					{
						//goto line must have number
						if (processedLine.Length < 3)
							throw new ProgramException("Invalid Syntax", originalLineNumber);
						if (compile)
						{
							//in compile mode it just checks if the value is a valid number
							int gotoLine;
							if (!int.TryParse(processedLine[2], out gotoLine))
							{
								throw new ProgramException($"Cannot Parse Goto Line: {gotoLine}", originalLineNumber);
							}
						} //it can be assumed by runtime that its valid so there is no try/parse line
						else
							i = Convert.ToInt32(processedLine[2], 10) - 1; //i is the current line number
					}
					else //continue or break statement 
					{
						if (processedLine.Length < 5)
							throw new ProgramException("Invalid Syntax", originalLineNumber);

						//format goes 'continue goto *linenum*'
						if (processedLine[2].Equals("goto"))
						{
							if (compile)
							{
								int gotoLine;
								if (!int.TryParse(processedLine[4], out gotoLine))
								{
									throw new ProgramException($"Cannot Parse Goto Line: {gotoLine}", originalLineNumber);
								}
							}
							else
							{
								i = Convert.ToInt32(processedLine[4], 10) - 1;
								//if the statement is break, a flag is used to exit current loop
								if (processedLine[0].Equals("break"))
									breakFlag = true;
							}
						}
					}
					if (!processedLine[0].Equals("break"))
					{
						//break statements dont remove variable here because they are ultimately followed by a goto statement for exiting loop which will land here
						List<string> remove = varStack.Pop();
						foreach (string var in remove)
						{
							vars.Remove(var);
						}
					}
				}
				/*
				 * This chunk of code is for processing declaration of array types
				 * The syntax for array declaration is either
				 * int[] arr = {0,2,4,6,8};
				 * or
				 * int[] arr = new int[5];
				 * No multidimensional arrays are allowed
				 */
				else if (processedLine[0].Equals("int[]"))//declaration of int array
				{
					List<int> array = new List<int>();
					if (processedLine[1].Equals("=")) throw new ProgramException("Variable Name Required", originalLineNumber);
					string varName = processedLine[2];
					//code for declaring new array
					if (processedLine.Length > 2 && processedLine[4].Equals("=") && processedLine[6].Equals("{"))
					{
						bool ended = false;
						for (int j = 7; j < processedLine.Length; j++)
						{
							//currently you cant put statements in the array declaration
							if (processedLine[j].Equals("}"))
							{
								ended = true;
								break;
							}
							else if (processedLine[j].Equals(","))
							{
								//currently commas are ignored
								continue;
							}
							else
							{
								array.Add((int)Evaluate(processedLine[j], originalLineNumber, compile));
							}
						}
						if (!ended) throw new ProgramException("Array Requires Ending Brace", originalLineNumber);
						else
						{
							vars.Add(varName, array.ToArray());
							varStack.Peek().Add(varName);
						}
					}
					else
					{
						//This code is for declaring an empty array of a given length

						if (processedLine.Length > 8 && processedLine[4].Equals("=") && processedLine[6].Equals("new") && processedLine[8].Contains("int["))
						{
							int end = processedLine[8].IndexOf("]");
							if (end == -1) throw new ProgramException("Invalid Array Syntax", originalLineNumber);
							int arrayLength = (int)Evaluate(processedLine[8].Substring(4, end), originalLineNumber, compile);
							vars.Add(varName, new int[arrayLength]);
							varStack.Peek().Add(varName);
						}
						else
						{
							throw new ProgramException("Invalid Array Syntax", originalLineNumber);
						}
					}
				}
				else if (processedLine[0].Equals("string[]"))
				{
					//Same code for string arrays, not a useful data type but added in for consistency
					List<string> array = new List<string>();
					if (processedLine[2].Equals("=")) throw new ProgramException("Variable Name Required", originalLineNumber);
					string varName = processedLine[2];
					if (processedLine.Length > 7 && processedLine[4].Equals("=") && processedLine[6].Equals("{"))
					{
						bool ended = false;
						for (int j = 7; j < processedLine.Length; j++)
						{

							if (processedLine[j].Equals("}"))
							{
								ended = true;
								break;
							}
							else if (processedLine[j].Equals(","))
							{
								continue;
							}
							else
							{ //string evalutation, fairly straightforward
								array.Add((string)Evaluate(processedLine[j], originalLineNumber, compile));
							}
						}
						if (!ended) throw new ProgramException("Array Requires Ending Brace", originalLineNumber);
						else
						{
							vars.Add(varName, array.ToArray());
							varStack.Peek().Add(varName);
						}
					}
					else
					{
						if (processedLine.Length > 8 && processedLine[4].Equals("=") && processedLine[6].Equals("new") && processedLine[8].Contains("string["))
						{
							//Code for adding an empty array of strings, c# is dynamic
							//Exists for uniformity rather than practicality
							int end = processedLine[8].IndexOf("]");
							if (end == -1) throw new ProgramException("Invalid Array Syntax", originalLineNumber);
							int arrayLength = (int)Evaluate(processedLine[8].Substring(7, end), originalLineNumber, compile);
							vars.Add(varName, new string[arrayLength]);
							varStack.Peek().Add(varName);
						}
						else
						{
							throw new ProgramException("Invalid Array Syntax", originalLineNumber);
						}
					}
				}
				else if (processedLine[0].Equals("double[]"))
				{
					//Parsing array of doubles or floats
					//Same kind of code as the int and string array
					List<double> array = new List<double>();
					if (processedLine[2].Equals("=")) throw new ProgramException("Variable Name Required", originalLineNumber);
					string varName = processedLine[2];
					if (processedLine.Length > 8 && processedLine[4].Equals("=") && processedLine[6].Equals("{"))
					{
						bool ended = false;
						for (int j = 7; j < processedLine.Length; j++)
						{

							if (processedLine[j].Equals("}"))
							{
								ended = true;
								break;
							}
							else if (processedLine[j].Equals(","))
							{
								continue;
							}
							else
							{
								array.Add((double)Evaluate(processedLine[j], originalLineNumber, compile));
							}
						}
						if (!ended) throw new ProgramException("Array Requires Ending Brace", originalLineNumber);
						else
						{
							vars.Add(varName, array.ToArray());
							varStack.Peek().Add(varName);
						}
					}
					else
					{
						if (processedLine.Length > 7 && processedLine[4].Equals("=") && processedLine[6].Equals("new") && processedLine[8].Contains("double["))
						{
							//Code for empty array of doubles, all 0.0
							int end = processedLine[8].IndexOf("]");
							if (end == -1) throw new ProgramException("Invalid Array Syntax", originalLineNumber);
							int arrayLength = (int)Evaluate(processedLine[8].Substring(7, end), originalLineNumber, compile);
							vars.Add(varName, new double[arrayLength]);
							varStack.Peek().Add(varName);
						}
						else
						{
							throw new ProgramException("Invalid Array Syntax", originalLineNumber);
						}
					}
				}
				else if (processedLine[0].Equals("bool[]"))
				{
					//Code for processing bool array, similar use to the string array
					List<bool> array = new List<bool>();
					if (processedLine[2].Equals("=")) throw new ProgramException("Variable Name Required", originalLineNumber);
					string varName = processedLine[2];
					if (processedLine.Length > 7 && processedLine[4].Equals("=") && processedLine[6].Equals("{"))
					{
						bool ended = false;
						for (int j = 7; j < processedLine.Length; j++)
						{

							if (processedLine[j].Equals("}"))
							{
								ended = true;
								break;
							}
							else if (processedLine[j].Equals(","))
							{
								continue;
							}
							else
							{
								array.Add((bool)Evaluate(processedLine[j], originalLineNumber, compile));
							}
						}
						if (!ended) throw new ProgramException("Array Requires Ending Brace", originalLineNumber);
						else
						{
							vars.Add(varName, array.ToArray());
							varStack.Peek().Add(varName);
						}
					}
					else
					{
						if (processedLine.Length > 7 && processedLine[4].Equals("=") && processedLine[6].Equals("new") && processedLine[8].Contains("bool["))
						{
							int end = processedLine[8].IndexOf("]");
							if (end == -1) throw new ProgramException("Invalid Array Syntax", originalLineNumber);
							int arrayLength = (int)Evaluate(processedLine[8].Substring(5, end), originalLineNumber, compile);
							vars.Add(varName, new bool[arrayLength]);
							varStack.Peek().Add(varName);
						}
						else
						{
							throw new ProgramException("Invalid Array Syntax", originalLineNumber);
						}
					}
				}
				else if (processedLine[0].Equals("int"))
				{
					//int variable declarition code
					if (processedLine[2].Equals("="))
					{
						//Cant have lines such as 
						//int = 35;
						//Must declare a name for the variable
						throw new ProgramException("Variable Name Required", originalLineNumber);
					}
					else
					{
						string varName = processedLine[2];
						if (processedLine.Length > 5 && processedLine[4].Equals("="))
						{
							//assignExpr is the string representation of the number expression being assigned
							//
							string assignExpr = "";
							for (int j = 5; j < processedLine.Length; j++)
							{
								//incomplete implementation of multiple statements on one line
								//currently just stops after seeing a comma instance
								if (processedLine[j].Equals(","))
									break;
								else
									assignExpr += processedLine[j];
							}
							object val = Evaluate(assignExpr, originalLineNumber, compile);
							//here it trims doubles if assigned
							if (val is int || val is double)
							{
								vars.Add(varName, (int)val);
								varStack.Peek().Add(varName);
							}
							else
							{
								throw new ProgramException("Invalid Data Type for Integer Assignment", originalLineNumber);
							}
						} //Here is the code for variable declarations with no assignment
						else
						{
							vars.Add(varName, 0);
							varStack.Peek().Add(varName);
						}

					}
				}
				else if (line.StartsWith("string"))
				{
					/*Code for declaring a string variable
					 * Currently doesn't have support for variable insertions $"Value: {variable}" 
					 * Although you can add strings with any datatype and it will remain a string
					 */
					if (processedLine[2].Equals("="))
					{
						throw new ProgramException("Variable Name Required", originalLineNumber);
					}
					else
					{
						//Here the code for declaration is fairly redundant
						//It could be condensed, but in the interest of readability and diagnosing possible errors it is left this way.
						//
						string varName = processedLine[2];
						if (processedLine.Length > 5 && processedLine[4].Equals("="))
						{
							string assignExpr = "";
							for (int j = 5; j < processedLine.Length; j++)
							{
								if (processedLine[j].Equals(","))
									break;
								else
									assignExpr += processedLine[j];
							}
							object val = Evaluate(assignExpr, originalLineNumber, compile);
							if (val is string)
							{
								vars.Add(varName, (string)val);
								varStack.Peek().Add(varName);
							}
							else
							{
								throw new ProgramException("Invalid Data Type for String Assignment", originalLineNumber);
							}

						}
						else
						{
							vars.Add(varName, "");
							varStack.Peek().Add(varName);
						}
					}
				}
				else if (line.StartsWith("double"))
				{
					if (processedLine[2].Equals("="))
					{
						throw new ProgramException("Variable Name Required", originalLineNumber);
					}
					else
					{
						string varName = processedLine[2];
						if (processedLine.Length > 2 && processedLine[4].Equals("="))
						{
							string assign = "";
							for (int j = 5; j < processedLine.Length; j++)
							{
								if (processedLine[j].Equals(","))
									break;
								else
									assign += processedLine[j];
							}
							object val = Evaluate(assign, originalLineNumber, compile);
							if (val is int || val is double)
							{
								vars.Add(varName, (double)val);
								varStack.Peek().Add(varName);
							}
							else
							{
								throw new ProgramException("Invalid Data Type for Double Assignment", originalLineNumber);
							}

						}
						else
						{
							vars.Add(varName, 0.0);
							varStack.Peek().Add(varName);
						}
					}
				}
				else if (line.StartsWith("bool"))
				{
					if (processedLine[2].Equals("="))
					{
						throw new ProgramException("Variable Name Required");
					}
					else
					{
						string varName = processedLine[2];
						if (processedLine.Length > 5 && processedLine[4].Equals("="))
						{
							string assign = "";
							for (int j = 5; j < processedLine.Length; j++)
							{
								if (processedLine[j].Equals(","))
									break;
								else
									assign += processedLine[j];
							}
							object val = Evaluate(assign, originalLineNumber, compile);
							if (val is bool)
							{
								vars.Add(varName, (bool)val);
								varStack.Peek().Add(varName);
							}
							else
							{
								throw new ProgramException("Invalid Data Type for Boolean Assignment", originalLineNumber);
							}

						}
						else
						{
							vars.Add(varName, false);
							varStack.Peek().Add(varName);
						}
					}
				}
				else
				{
					//This is code for reassigning a variable
					object var;
					string varName = processedLine[0];
					//If no assignment takes place, assumed to be function call
					if (processedLine.Length > 2 && !processedLine[2].Equals("="))
					{
						Evaluate(line, originalLineNumber, compile);
						continue;
					}
					//Check if variable has been declared, if not throw error
					if (!vars.TryGetValue(varName, out var))
						throw new ProgramException("Variable Does Not Exist", originalLineNumber);
					if (processedLine.Length > 4)
					{
						string assign = "";
						for (int j = 4; j < processedLine.Length; j++)
						{
							if (processedLine[j].Equals(","))
								break;
							else
								assign += processedLine[j];
						}
						object val = Evaluate(assign, originalLineNumber, compile);
						vars[varName] = val;
					}
				}
			}//End of program the variable stack is cleared along with all saved values
			vars.Clear();
			varStack.Clear();
		}

		//This method takes care of the initial syntactic analysis
		//Here program lines are split into tokens
		//The corresponding line number is preserved
		private List<(int, string)> TokenizeProgram(string[] program)
		{
			List<(int, string)> tokenizedProgram = new List<(int, string)>();
			//Variable tracks original line number
			int lineNum = 0;
			foreach (string line in program)
			{
				//Parentheses,Commas,Semicolons,Braces and Assignment Operators are split apart
				//Empty spaces are removed as well

				string[] processedLine = Regex.Split(line.TrimStart().TrimEnd(), @"([(),;{}=])");
				//Takes empty strings out of the list
				processedLine = processedLine.Where(x => !string.IsNullOrEmpty(x)).ToArray();
				if (processedLine.Length == 0) continue;
				/*This code checks for valid terminal character
				 * Rolls out increment and decrement operation
				 */
				if (processedLine[processedLine.Length - 1].Contains("}") || processedLine[processedLine.Length - 1].Contains("{") || processedLine[processedLine.Length - 1].Contains(";") || processedLine[processedLine.Length - 1].Contains(",")
					|| processedLine[0].Contains(ifStat) || processedLine[0].Contains(forStat) || processedLine[0].Contains(whileStat))
				{
					foreach (string token in processedLine)
					{
						if (token.Contains(incr) || token.Contains(decr))
						{
							string tokenCopy = token;
							//Assume its possible that there are multiple incr/decr operations in a given line
							//Can't vouch this works 100% of the time, but its been fairly tested
							while (tokenCopy.Contains(incr) || tokenCopy.Contains(decr))
							{
								int incrIndex = tokenCopy.IndexOf(incr);
								int decrIndex = tokenCopy.IndexOf(decr);
								if ((incrIndex < decrIndex || decrIndex == -1) && incrIndex != -1)
								{
									tokenizedProgram.Add((lineNum, tokenCopy.Substring(0, incrIndex)));
									tokenizedProgram.Add((lineNum, tokenCopy.Substring(incrIndex, incr.Length)));
									tokenCopy = tokenCopy.Substring(incrIndex + incr.Length);
									continue;
								}
								if ((incrIndex > decrIndex || incrIndex == -1) && decrIndex != -1)
								{
									tokenizedProgram.Add((lineNum, tokenCopy.Substring(0, decrIndex)));
									tokenizedProgram.Add((lineNum, tokenCopy.Substring(decrIndex, incr.Length)));
									tokenCopy = tokenCopy.Substring(decrIndex + decr.Length);
									continue;
								}
							}
							tokenizedProgram.Add((lineNum, tokenCopy));
							continue;
						}

						if (!token.Equals(""))
							tokenizedProgram.Add((lineNum, token));
					}
					//next line in program
					lineNum++;
				}
				else
				{
					throw new ProgramException("Missing Terminating Character", lineNum);
				}

			}
			return tokenizedProgram;
		}

		/* Lexical Analysis is the initial run through of the code
		* Here loops and conditional statements are decomposed into a more simple routine consisting if(boolean statement) goto line#
		* All pre/post increment and decrement statements are expanded out in a order consistent with their meaning
		* This turns the code into all executable statements, the output will be assignments, conditional gotos, function calls
		* The correct function and variables of the code remain
		* Recursive method so that loops can be evaluated
		* 
		* Takes in the original tokenization list and original line numbers
		* The starting line number of the code, helps keep track of which line in the program goto statements will flow to
		* break line number parameter is for when a break statment is implemented
		* continue statement is the update that normally happens at the end of the loop
		* breakOriginalLineNum is just tracking the original line number incase of errors
		*/

		private List<(int, string)> LexicalAnalysis(List<(int, string)> tokens, int startLineNum = 0, int breakLineNum = 0, string continueStatement = "", int breakOriginalLineNum = 0)
		{
			//remove empty lines

			tokens = tokens.Where(x => !x.Item2.Equals("")).ToList();
			List<(int, string)> parsedProgram = new List<(int, string)>();
			string parsedLine = "";
			for (int i = 0; i < tokens.Count; i++)
			{
				//Queue holds any post increment/decrement as they are added, everytime a line is added, the queue is emptied if there are any post increment statements 
				Queue<string> incr_decr_lines = new Queue<string>();
				(int programLine, string elem) = tokens[i];
				elem = elem.TrimEnd();
				if (elem.Equals("for"))
				{
					//Determines the elongated code for a for loop
					//Adds a declaration line
					i++;
					(programLine, elem) = tokens[i];
					//start with interpreting the for loop
					if (!elem.Equals("("))
					{
						//if there is no immediate parentheses then sytax is incorrect
						throw new ProgramException("Error Invalid Syntax: " + programLine);
					}
					else
					{
						List<(int, string)> declaration = new List<(int, string)>();
						//parse the initial declaration and add it to its own line
						while (!elem.Equals(";"))
						{
							i++;
							(programLine, elem) = tokens[i];
							declaration.Add((programLine, elem));
							if (elem.Equals(")"))
							{
								throw new ProgramException("Error Invalid Syntax", programLine);
							}
						}
						//The declaration is run through parseLines for safety reasons
						List<(int, string)> forDeclaration = LexicalAnalysis(declaration, startLineNum, breakLineNum);
						//This the declaration is added to the program
						foreach ((int blockNum, string blockLine) in forDeclaration)
						{
							parsedProgram.Add((blockNum, blockLine));
							startLineNum++;
						}
						//parse the condition statement
						string ifStatement = "if(";
						i++;
						(programLine, elem) = tokens[i];
						while (!elem.Equals(";"))
						{
							ifStatement += elem;
							i++;
							(programLine, elem) = tokens[i];
							if (elem.Equals(")"))
							{
								throw new ProgramException("Error Invalid Syntax", programLine);
							}
						}
						//else goto will be the exit out of the for loop
						ifStatement += ") else goto ";
						i++;
						(programLine, elem) = tokens[i];
						string updateStatement = "";
						//updateLine will be the usual increment/decrement or adjustment of the declaration/condition variable
						List<(int, string)> updateCodeBlock = new List<(int, string)>();
						while (!elem.Equals(")"))
						{
							updateCodeBlock.Add((programLine, elem));
							i++;
							(programLine, elem) = tokens[i];
							if (elem.Equals(";"))
							{
								throw new ProgramException("Error Invalid Syntax", programLine);
							}
						}
						//add a semicolon so i can run the line through parseLines
						updateCodeBlock.Add((programLine, ";"));
						//currently it just takes the first element, this might be updated later
						updateStatement = LexicalAnalysis(updateCodeBlock, startLineNum, breakLineNum).ToArray()[0].Item2;
						i++;
						int forLineNum = programLine;
						(programLine, elem) = tokens[i];
						//Main idea here is to put the block into its own list and run it as a subprogram
						//Therefore if there are other loops within, those get processed correctly
						//Its important that the correct line numbers are passed
						List<(int, string)> sublist = new List<(int, string)>();
						//Two scenerios here, there is block of code or it is a one line for loop
						if (elem.Equals("{"))
						{
							//Stack tracks if block has ended
							Stack<string> block = new Stack<string>();
							block.Push(elem);
							while (block.Count != 0)
							{
								//goes through elements
								i++;
								(int lineNum, string lineElem) = tokens[i];
								if (lineElem.Equals("{"))
								{
									block.Push(lineElem);
								}
								if (lineElem.Equals("}"))
								{
									block.Pop();
								}//adds all elements unless block is empty
								if (block.Count != 0)
									sublist.Add((lineNum, lineElem));
							}
						}
						else
						{
							//Grabs until nearest endline ;
							sublist.Add((programLine, elem));
							do
							{
								i++;
								(programLine, elem) = tokens[i];
								sublist.Add((programLine, elem));
							} while (!elem.Equals(";"));
						}
						int parsedLineNum = startLineNum;
						//sublist here is parsed, pass in the update at the end of the for loop and the line it should go back to at end of block
						//updateLine and forLineNum are passed in in case of continue/break
						List<(int, string)> parsedSubProgram = LexicalAnalysis(sublist, startLineNum + 1, startLineNum, updateStatement, forLineNum);
						//Here the statements are added because we know how long the subprogram is now
						//So exit statements will take to the correct line
						ifStatement += (startLineNum + parsedSubProgram.Count + 3); //This will be the first line number outside of the loop
						parsedProgram.Add((forLineNum, ifStatement));
						startLineNum++;
						//add all the subprogram
						foreach ((int blockNum, string blockLine) in parsedSubProgram)
						{
							parsedProgram.Add((blockNum, blockLine));
							startLineNum++;
						}
						//updateStatement is added
						parsedProgram.Add((forLineNum, updateStatement));
						startLineNum++;
						//Here we go back to the if statement 
						parsedProgram.Add((forLineNum, "goto " + parsedLineNum));
						startLineNum++;
						parsedLine = "";
					}
				}
				else if (elem.Equals("while"))
				{
					//Determines the elongated code for a while loop
					i++;
					(programLine, elem) = tokens[i];
					//start with interpreting the for loop
					if (elem.Equals("("))
					{
						//parse the condition

						string ifStatement = "if(";
						i++;
						(programLine, elem) = tokens[i];
						while (!elem.Equals(")"))
						{
							ifStatement += elem;
							i++;
							(programLine, elem) = tokens[i];
						}
						//if the condition is evaluated to be false it needs to know which line to goto
						ifStatement += ")else goto ";
						i++;
						(programLine, elem) = tokens[i];
						int whileLineNum = programLine;
						List<(int, string)> sublist = new List<(int, string)>();
						//Similar to the for loop, the subprogram gets added to a sublist and then evaluated
						if (elem.Equals("{"))
						{
							Stack<string> block = new Stack<string>();
							block.Push(elem);
							while (block.Count != 0)
							{
								i++;
								(int lineNum, string lineElem) = tokens[i];
								if (lineElem.Equals("{"))
								{
									block.Push(lineElem);
								}
								if (lineElem.Equals("}"))
								{
									block.Pop();
								}
								if (block.Count != 0)
									sublist.Add((lineNum, lineElem));
							}
						}
						else
						{
							//for one line while statements
							sublist.Add((programLine, elem));
							do
							{
								i++;
								(programLine, elem) = tokens[i];
								sublist.Add((programLine, elem));
							} while (!elem.Equals(";"));
						}
						int parsedLineNum = startLineNum;
						//run through, there is no updateLine here
						List<(int, string)> parsedSubProgram = LexicalAnalysis(sublist, startLineNum + 1, startLineNum);
						ifStatement += (startLineNum + parsedSubProgram.Count + 2);
						parsedProgram.Add((whileLineNum, ifStatement));
						startLineNum++;
						foreach ((int blockNum, string blockLine) in parsedSubProgram)
						{
							parsedProgram.Add((blockNum, blockLine));
							startLineNum++;
						}
						parsedProgram.Add((whileLineNum, "goto " + parsedLineNum));
						startLineNum++;
						parsedLine = "";
					}
					else if (elem.Equals("}"))
					{
						//Can't remember exactly why this condition is here, might be because elem is the final } in some cases
						//Marking this so it can be changed later
						continue;
					}
					else
					{
						throw new ProgramException("Error Invalid Syntax", programLine);
					}
				}
				else if (elem.Equals("if"))
				{
					//if statements parsed here
					i++;
					(programLine, elem) = tokens[i];
					if (!elem.Equals("("))
					{
						throw new ProgramException("Error Invalid Syntax", programLine);
					}
					else
					{
						int ifLine = programLine;
						parsedLine = "if";
						while (!elem.Equals(")"))
						{
							parsedLine += elem;
							i++;
							(programLine, elem) = tokens[i];
						}
						//make sure if condition is not empty
						if (parsedLine.Equals("")) throw new ProgramException("Error Invalid Syntax", programLine);
						parsedLine += ") else goto ";
						string ifStatement = parsedLine;
						parsedLine = "";
						i++;
						(programLine, elem) = tokens[i];
						//Similar idea to the for and while loops, you have a subprogram parsed and added in
						List<(int, string)> sublist = new List<(int, string)>();
						if (elem.Equals("{"))
						{
							Stack<string> block = new Stack<string>();
							block.Push(elem);
							while (block.Count != 0)
							{
								i++;
								(int lineNum, string lineElem) = tokens[i];
								if (lineElem.Equals("{"))
								{
									block.Push(lineElem);
								}
								if (lineElem.Equals("}"))
								{
									block.Pop();
								}
								if (block.Count != 0)
									sublist.Add((lineNum, lineElem));
							}
						}
						else
						{

							sublist.Add((programLine, elem));
							do
							{
								i++;
								(programLine, elem) = tokens[i];
								sublist.Add((programLine, elem));
							} while (!elem.Equals(";"));
						}
						int parsedLineNum = startLineNum;
						//Parameters are passed back in because you could be in the middle of a loop
						List<(int, string)> parsedIfSubProgram = LexicalAnalysis(sublist, startLineNum + 1, breakLineNum, continueStatement, breakOriginalLineNum);
						if (i < tokens.Count - 1)
						{
							i++;
							(programLine, elem) = tokens[i];
						}
						else elem = "";
						//Here the else block is checked and added another sublist
						if (elem.Equals("else"))
						{
							i++;
							(programLine, elem) = tokens[i];
							sublist = new List<(int, string)>();
							if (elem.Equals("{"))
							{
								Stack<string> block = new Stack<string>();
								block.Push(elem);
								while (block.Count != 0)
								{
									i++;
									(int lineNum, string lineElem) = tokens[i];
									if (lineElem.Equals("{"))
									{
										block.Push(lineElem);
									}
									if (lineElem.Equals("}"))
									{
										block.Pop();
									}
									if (block.Count != 0)
										sublist.Add((lineNum, lineElem));
								}
							}
							else
							{

								sublist.Add((programLine, elem));
								do
								{
									i++;
									(programLine, elem) = tokens[i];
									sublist.Add((programLine, elem));
								} while (!elem.Equals(";"));
							}
							//The else block gets parsed and put into its own list
							int parsedLineNum2 = startLineNum;
							List<(int, string)> parsedElseSubProgram = LexicalAnalysis(sublist, startLineNum + parsedIfSubProgram.Count + 2, breakLineNum, continueStatement, breakOriginalLineNum);
							ifStatement += (startLineNum + parsedIfSubProgram.Count + 1);
							parsedProgram.Add((ifLine, ifStatement));
							startLineNum++;
							//add in the if block
							foreach ((int blockNum, string blockLine) in parsedIfSubProgram)
							{
								parsedProgram.Add((blockNum, blockLine));
								startLineNum++;
							}
							//we know now how long the else is, so the goto can be configured
							parsedProgram.Add((ifLine, "goto " + (startLineNum + parsedElseSubProgram.Count + 2)));
							startLineNum++;
							foreach ((int blockNum, string blockLine) in parsedElseSubProgram)
							{
								parsedProgram.Add((blockNum, blockLine));
								startLineNum++;
							}
							parsedLine = "";
						} // if there is no else statement then we just add the if block
						else
						{
							//just exits out length of block
							ifStatement += (startLineNum + parsedIfSubProgram.Count + 1);
							parsedProgram.Add((ifLine, ifStatement));
							startLineNum++;
							foreach ((int blockNum, string blockLine) in parsedIfSubProgram)
							{
								parsedProgram.Add((blockNum, blockLine));
								startLineNum++;
							}
							parsedLine = "";
							//Have to go back an element
							i--;
						}
					}
				}
				else if (elem.Equals("break") || elem.Equals("continue"))
				{
					//for break or continue statements we can add the correct exit statements
					if (!continueStatement.Equals("") && elem.Equals("continue"))
					{
						//when continue is called we still need to update in for loop
						parsedProgram.Add((breakOriginalLineNum, continueStatement));
						startLineNum++;
					}
					parsedLine = elem;
					parsedLine += " goto ";
					parsedLine += breakLineNum;
					//go to top of loop and then there will be a break flag to void the condition
					parsedProgram.Add((programLine, parsedLine));
					startLineNum++;
					parsedLine = "";
				}
				else
				{
					//This part of the code deals with the non-conditional statements
					//Here is where some early preprocessing is done
					while (i < tokens.Count - 1 && !elem.Equals(";"))
					{
						//expand out pre increment and decrement statements
						if (elem.Equals("++"))
						{
							char nextChar = tokens[i + 1].Item2[0];
							if (Regex.IsMatch(nextChar.ToString(), "[a-z]", RegexOptions.IgnoreCase))
							{ //pre increment
								i++;
								(programLine, elem) = tokens[i];
								parsedProgram.Add((programLine, elem + " = " + elem + " + 1"));
								//These get added before the current line in the order they are declared
								startLineNum++;
							}
						}
						if (elem.Equals("--"))
						{ //pre decrement
							char nextChar = tokens[i + 1].Item2[0];
							if (Regex.IsMatch(nextChar.ToString(), "[a-z]", RegexOptions.IgnoreCase))
							{
								i++;
								(programLine, elem) = tokens[i];
								parsedProgram.Add((programLine, elem + " = " + elem + " - 1"));
								startLineNum++;
							}
						}

						char start = elem.TrimStart(' ')[0];
						//post increment is checked here, looks for variable alongside increment notation
						if (Regex.IsMatch(start.ToString(), "[a-z]", RegexOptions.IgnoreCase) && i < tokens.Count - 1 && tokens[i + 1].Item2.Equals("++"))
						{
							//adds the line to a queue while the current variable is 
							incr_decr_lines.Enqueue(elem.TrimStart(' ') + " = " + elem.TrimStart(' ') + " + 1");
							if (!parsedLine.Equals(""))
								parsedLine += elem;
							//resume parsing
							i = i + 2;
							(programLine, elem) = tokens[i];
						}
						else if (Regex.IsMatch(start.ToString(), "[a-z]", RegexOptions.IgnoreCase) && i < tokens.Count - 1 && tokens[i + 1].Item2.Equals("--"))
						{
							//post decrement
							incr_decr_lines.Enqueue(elem.TrimStart(' ') + " = " + elem.TrimStart(' ') + " - 1");
							if (!parsedLine.Equals(""))
								parsedLine += elem;
							i = i + 2;
							(programLine, elem) = tokens[i];
						}
						else
						{ //generic add element to line
							parsedLine += elem;
							i++;
							(programLine, elem) = tokens[i];
						}
						if (!parsedLine.EndsWith(" "))
						{
							if (elem.Equals("="))
							{
								//Augmented Assignments parsing here, just expands out the line
								if (parsedLine.EndsWith("+"))
								{
									//messing with the whitespace so it is consistent
									parsedLine = parsedLine.TrimEnd('+').TrimEnd(' ') + " " + elem + " " + parsedLine.TrimEnd('+').TrimEnd(' ') + " + ";
									i++;
									(programLine, elem) = tokens[i];
									continue;
								}
								else if (parsedLine.EndsWith("-"))
								{
									parsedLine = parsedLine.TrimEnd('-').TrimEnd(' ') + " " + elem + " " + parsedLine.TrimEnd('-').TrimEnd(' ') + " - ";
									i++;
									(programLine, elem) = tokens[i];
									continue;
								}
								else if (parsedLine.EndsWith("*"))
								{
									parsedLine = parsedLine.TrimEnd('*').TrimEnd(' ') + " " + elem + " " + parsedLine.TrimEnd('*').TrimEnd(' ') + " * ";
									i++;
									(programLine, elem) = tokens[i];
									continue;
								}
								else if (parsedLine.EndsWith("/"))
								{
									parsedLine = parsedLine.TrimEnd('/').TrimEnd(' ') + " " + elem + " " + parsedLine.TrimEnd('/').TrimEnd(' ') + " / ";
									i++;
									(programLine, elem) = tokens[i];
									continue;
								}
								else elem = " " + elem; //adds space as needed

							}
						}
						if (parsedLine.EndsWith("="))
						{// Might be redundant but checks spacing
							if (!elem.StartsWith(" "))
							{
								parsedLine += " ";
							}
						}
						if (elem.StartsWith("0x"))
						{ //convert hex elements to base 10
							elem = Convert.ToInt32(elem, 16).ToString();
						}
						if (elem.StartsWith("0b"))
						{ //convert binary elements to base 10
							elem = Convert.ToInt32(elem.Substring(2), 2).ToString();
						}

					}
					if (!parsedLine.Equals(""))
					{
						//add the line
						parsedProgram.Add((programLine, parsedLine));
						startLineNum++;
						parsedLine = "";
					}
					while (incr_decr_lines.Count != 0)
					{
						//add all post increments
						parsedProgram.Add((programLine, incr_decr_lines.Dequeue()));
						startLineNum++;
					}
				}
			}
			//return the parsed out program
			return parsedProgram;
		}






		/* Evaluate is an important method because it evaluates all arithmetic and calls methods
		 * It is inheriently recursive as content in parentheses needs to be evaluated to maintain order of operations
		 * 
		 */


		public object Evaluate(string evalLine,int originalLineNumber,bool compile = false)
		{
			//The highest priority goes to method calls
			//Stack will hold the value returned from method
			Stack<object> stack = new Stack<object>();
			evalLine = evalLine.TrimStart(' ');
			//Check if line begins with method call
			if (methods.ToArray().Any(evalLine.StartsWith))
			{	//split spaces, commas and parentheses
				string[] preprocessedLine = Regex.Split(evalLine, @"([(), ])");
				//make sure strings dont get split up
				string[] processedLine = StitchSplitStrings(preprocessedLine);
				string inputs = "";
				if (processedLine[1] == "(")
				{
					//Evaluate all parameters
					List<string> expressions = new List<string>();
					//stack keeps track if end of parameters have been reached
					Stack<string> parents = new Stack<string>();
					int tracker = processedLine[0].Length + processedLine[1].Length;
					for (int j = 2; j < processedLine.Length; j++)
					{
						//Tracker for index in line for evaluating parentheses in parameters
						tracker += processedLine[j].Length;
						if (methods.ToArray().Any(processedLine[j].Contains))
						{
							//any parentheses within parameters need to be evaluated
							string subFunction = processedLine[j];
							Stack<string> subParents = new Stack<string>();
							j++;
							do
							{
								if (processedLine[j].Equals("("))
								{
									subParents.Push(processedLine[j]);
								}
								if (processedLine[j].Equals(")"))
								{
									subParents.Pop();
								}
								subFunction += processedLine[j];
								j++;
								tracker += processedLine[j].Length;
							} while (subParents.Count != 0);
							//add evaluated parentheses back to the line
							inputs += Evaluate(subFunction, originalLineNumber, compile);
							j--;
						}
						else if (processedLine[j].Equals(","))
						{
							//Expressions holds each parameter
							//inputs string is reset
							expressions.Add(inputs);
							inputs = "";
						}
						else if (processedLine[j].Equals(")"))
						{
							//if all parentheses have been closed, then the parameters can be evaluated
							if (parents.Count == 0)
							{
								expressions.Add(inputs);
								inputs = "";
								evalLine = evalLine.Substring(tracker);
								break;
							}
							else
							{
								parents.Pop();
							}
						}
						else if (processedLine[j].Equals("("))
						{
							parents.Push(processedLine[j]);
						}
						else
							inputs += processedLine[j];
					}
					//evaluated parameters goes into a list
					List<object> evaluatedParameters = new List<object>();
					foreach (string expr in expressions) evaluatedParameters.Add(Evaluate(expr,originalLineNumber,compile));
					// switch case statement for which method is called
					switch (processedLine[0])
					{
						case "Read":
							if(evaluatedParameters.Count > 0)
							{   //Check parameters and then imput them into read function if not in compile mode
								if (evaluatedParameters[0] is string || evaluatedParameters[0] is bool)
									throw new ProgramException("Invalid Parameter Type",originalLineNumber);
								else if (compile)
								{
									//Compile mode just outputs a one from read
									//Zero could cause errors in odd cases
									stack.Push(1);
								}
								else
									stack.Push(Read((int)evaluatedParameters[0]));
							}
							break;
						case "Write":

							//Write method requires 2 parameters at least
							if (evaluatedParameters.Count > 1) { 
								if (evaluatedParameters[0] is string || evaluatedParameters[0] is bool || evaluatedParameters[1] is string || evaluatedParameters[1] is bool)
									throw new ProgramException("Invalid Parameter Type",originalLineNumber);
								else if (compile)
								{
									stack.Push(1);
								}
								else
									stack.Push(Write((int)evaluatedParameters[0],(int)evaluatedParameters[1]));
							}
							break;
						case "Delay":
							if (evaluatedParameters.Count > 0)
							{

								//Delay method requires one integer parameter in milliseconds
								if (evaluatedParameters[0] is string || evaluatedParameters[0] is bool)
									throw new ProgramException("Invalid Parameter Type",originalLineNumber);
								if(!compile)
									Delay((int)evaluatedParameters[0]);
							}
							break;
						case "Print":
							//Print function can have as many parameters as needed
							if(evaluatedParameters.Count > 1)
							{
								if(evaluatedParameters[0] is string)
								{
									//statements is the original string
									string statement = (string)evaluatedParameters[0];
									evaluatedParameters.RemoveAt(0);
									if(!compile)
										Print(statement, evaluatedParameters.ToArray());
								}
								else
								{
									throw new ProgramException("Invalid Parameter Format",originalLineNumber);
								}
							}
							else //if just one parameter, that is printed off when not in compile mode
							{
								if(!compile)
									Print(evaluatedParameters[0]);
							}
							break;
						case "PrintLine":
							//Same as print but moves to a new line
							if (evaluatedParameters.Count > 1)
							{
								if (evaluatedParameters[0] is string)
								{
									string statement = (string)evaluatedParameters[0];
									evaluatedParameters.RemoveAt(0);
									if(!compile)
										PrintLine(statement, evaluatedParameters.ToArray());
								}
							}
							else
							{
								if(!compile)
									PrintLine(evaluatedParameters[0]);
							}
							break;

					}

					//run each method here

				}
				else
				{
					throw new ProgramException("Invalid Syntax: Parentheses Required!",originalLineNumber);
				}
			}

			if (evalLine.Equals(""))
			{
				if(stack.Count != 0)
				{
					//if the string is empty and stack is populated, return its value
					//means the line was just a function call and not an assignment
					return stack.Pop();
				}
				else
				{
					//void function
					return 0;
				}
			}

			//Replaceing two character operators helps with further parsing
			//split the line into a list of tokens
			evalLine = evalLine.Replace("==", "#");
			evalLine = evalLine.Replace("!=", "~");
			evalLine = evalLine.Replace(">=", "@");
			evalLine = evalLine.Replace("<=", "$");

			List<string> infix = new List<string>(StitchSplitStrings(Regex.Split(evalLine, @"([()!=><+/*%#~$@ ])")));
			//This piece of code allows for spliting negative numbers and subtraction operation
			for(int i = 0; i < infix.Count; i++)
			{
				string operation = infix[i];
				if (isNumeric(operation) && operation.Substring(1).Contains("-"))
				{
					infix.RemoveAt(i);
					do
					{
						int index = operation.Substring(1).IndexOf("-");
						infix.Insert(i++, operation.Substring(0, index + 1));
						infix.Insert(i++, "-");
						operation = operation.Substring(index + 2);
					} while (isNumeric(operation) && operation.Substring(1).Contains("-"));
					infix.Insert(i++, operation);
				}
			}
			//Convert to prefix notation, allows for correct arithmetic
			List<string> prefix = PrefixConversion(infix);
			for(int i = prefix.Count -1 ; i >= 0; i--)
			{
				//prefix notation is evaluated using stack, operators push two elements off the stack and then push one onto it
				if (isOperator(prefix[i]))
				{
					object obj1 = stack.Pop();
					object obj2 = stack.Pop();
					try {
						//Call operator method on two objects
						switch (prefix[i]) {
							case "+":
								stack.Push(MathOperations.add(obj1, obj2));
								break;
							case "-":
								stack.Push(MathOperations.subtract(obj1, obj2));
								break;
							case "*":
								stack.Push(MathOperations.multiply(obj1, obj2));
								break;
							case "/":
								stack.Push(MathOperations.divide(obj1, obj2));
								break;
							case "^":
								stack.Push(MathOperations.power(obj1, obj2));
								break;
							case "#":
								stack.Push(MathOperations.EqualsCondition(obj1, obj2));
								break;
							case "~":
								stack.Push(MathOperations.NotEqualsCondition(obj1, obj2));
								break;
							case ">":
								stack.Push(MathOperations.GreaterThan(obj1, obj2));
								break;
							case "<":
								stack.Push(MathOperations.LessThan(obj1, obj2));
								break;
							case "@":
								stack.Push(MathOperations.GreaterThanOrEquals(obj1, obj2));
								break;
							case "$":
								stack.Push(MathOperations.LessThanOrEquals(obj1, obj2));
								break;
							default:
								throw new ProgramException("Error Invalid Operation!", originalLineNumber);
								//bogus error, should never be reached
						}
						
					}
					catch(MathOperations.OperationException ex)
					{
						//If operation has an error, the line number can traced
						throw new ProgramException(ex.Message, originalLineNumber);
					}
				}
				else
				{
					//elements get pushed onto stack
					stack.Push(ParseObject(prefix[i],originalLineNumber));
				}
			}
			//return last value after evaluation
			return stack.Pop();
		}
		//Parse object method is the part of the code that takes the string representation of data and guesses the datatype
		public object ParseObject(string value,int orignialLineNumber)
		{
			//starts with quotes, then parse it like a string
			if (value.StartsWith("\"") && value.EndsWith("\""))
				return value.Substring(1, value.Length - 2);
			else if (value.Equals("true")) //true and false are effectively keywords here
				return true;
			else if (value.Equals("false"))
				return false;
			else if (isNumeric(value) && value.Contains(".")) //if there is a decimal point
			{
				return double.Parse(value, CultureInfo.InvariantCulture);
			}
			else if (isNumeric(value))
			{//if not double then assume integer, hex or binary representation
				if (value.StartsWith("0x"))
					return Convert.ToInt32(value, 16);
				if (value.StartsWith("0b"))
					return Convert.ToInt32(value, 2);
				return int.Parse(value, CultureInfo.InvariantCulture);
			}
			else
			{	//array element value
				if (value.Contains("["))
				{
					int idx = value.IndexOf("[");
					int idx2 = value.IndexOf("]");
					string arrayIndex;
					if (idx2 == -1) throw new ProgramException("Array Index Requires Closing Bracket",orignialLineNumber);
					else
					{
						arrayIndex = value.Substring(idx+1, idx2 - (idx+1));
					}
					//grab array name, check if array exists
					string varName = value.Substring(0, idx);
					int parsedIndex = int.Parse(arrayIndex);
					object returnValue;
					bool success = vars.TryGetValue(varName, out returnValue);
					if (!success)
					{
						throw new ProgramException("Invalid Array Datatype", orignialLineNumber);
					}
					else{ //check if valid index then try and return value
						if (returnValue is int[]) { 
							if (parsedIndex < 0 || parsedIndex >= ((int[])returnValue).Length)
								throw new ProgramException("Invalid Array Index",orignialLineNumber);
							return ((int[])returnValue)[parsedIndex];
						}
						else if (returnValue is double[])
						{
							if (parsedIndex < 0 || parsedIndex >= ((double[])returnValue).Length)
								throw new ProgramException("Invalid Array Index",orignialLineNumber);
							return ((double[])returnValue)[parsedIndex];
						}
						else if (returnValue is string[])
						{
							if (parsedIndex < 0 || parsedIndex >= ((string[])returnValue).Length)
								throw new ProgramException("Invalid Array Index",orignialLineNumber);
							return ((string[])returnValue)[parsedIndex];
						}
						else if(returnValue is bool[])
						{
							if (parsedIndex < 0 || parsedIndex >= ((bool[])returnValue).Length)
								throw new ProgramException("Invalid Array Index",orignialLineNumber);
							return ((bool[])returnValue)[parsedIndex];
						}
						else
						{
							throw new ProgramException("Invalid Data Type", orignialLineNumber);
						}

					}
				}
				else { 
					object returnValue;
					bool success = vars.TryGetValue(value,out returnValue);
					if (success) return returnValue;
					else
					{
						throw new ProgramException("Invalid Variable Name", orignialLineNumber);
					}
				}
			}
		}
		/* Converts line to prefix notation
		 * It gives priority to certain operation in order to maintain order of operations
		 * 
		 */
		private List<string> PrefixConversion(List<string> infix)
		{
			//this method converts infix statements into prefix statements to avoid pemdas problems
			// stack for operators. 
			Stack<string> operators = new Stack<string>();
			// stack for operands. 
			Stack<string> operands = new Stack<string>();
			for (int i = 0; i < infix.Count; i++)
			{
				if (infix[i].Equals("") || infix[i].Equals(" ")) continue;
				// If current character is an 
				// opening bracket, then 
				// push into the operators stack. 
				if (infix[i].Equals("("))
				{
					operators.Push(infix[i]);
				}
				else if (infix[i].Equals(")"))
				{
					while ((operators.Count != 0) &&  !operators.Peek().Equals("("))
					{
						
						// operand 1 
						string op1 = operands.Pop();
						// operand 2 
						string op2 = operands.Pop();
						// operator 
						string op = operators.Pop();
						// Add operands and operator 
						// in form operator + 
						// operand1 + operand2. 
						string tmp = op+" " + op2+" " + op1;
						operands.Push(tmp);
					}// stack. 
					operators.Pop();
				}
				else if (!isOperator(infix[i]))
				{
					operands.Push(infix[i]);
				}
				else
				{
					while ((operators.Count!= 0) && (getPriority(infix[i]) <= getPriority(operators.Peek())))
					{
							string op1 = operands.Pop();

							string op2 = operands.Pop();

							string op = operators.Pop();

							string tmp = op + " " + op2 + " " + op1;
							operands.Push(tmp);
					}

						operators.Push(infix[i]);
					}
				}

				// Pop operators from operators stack 
				// until it is empty and add result 
				// of each pop operation in 
				// operands stack. 
				while (operators.Count != 0)
				{
					string op1 = operands.Pop();

					string op2 = operands.Pop();

					string op = operators.Pop();

					string tmp = op + " " + op2 + " " + op1;
					operands.Push(tmp);
				}

			// Final prefix expression is 
			// present in operands stack. 
			string result = operands.Pop();
			if (result.StartsWith("\"") && result.EndsWith("\""))
			{
				return new List<string> { result };
			}
			string[] processedLine = StitchSplitStrings(Regex.Split(result, @"([ ])"));
			return new List<string>(processedLine).Where(x => !x.Equals(" ")).ToList(); 
		}


		/* stitch split strings method is important because string may have elements that would get split apart leading to incomplete string elements
 * This method takes the split apart string array and looks for that happening, if it does it sticks the string back and returns a new string array
 * 
 */
		private string[] StitchSplitStrings(string[] line)
		{
			bool quote = false;
			string stitch = "";
			List<string> fixedLine = new List<string>();
			foreach (string elem in line)
			{
				if (quote)
				{   //if there is an identified unended quote, it stitches together until an ending quote appears
					if (elem.EndsWith("\""))
					{
						quote = false;
						stitch += elem;
						fixedLine.Add(stitch);
						stitch = "";
					}
					else
					{
						stitch += elem;
					}
				}
				else
				{//If the string starts with a " but doesnt have a an ending " we can assume it needs stitching
					if (elem.StartsWith("\"") && !elem.EndsWith("\""))
					{
						quote = true;
						stitch = elem;
					}
					else
					{ //otherwise keep the string array the same
						if (!elem.Equals("") && !elem.Equals(""))
							fixedLine.Add(elem);
					}
				}
			}
			return fixedLine.ToArray();
		}


		/* Helper method for prefixConversion
		 * Returns true if the string is one of the approved operators
		 */
		private bool isOperator(string c)
		{
				/*
				 * # is ==
				 * ~ is !=
				 * @ is >=
				 * $ is <=
				 */
				return c.Equals("+") || c.Equals("-") || c.Equals("*")
				|| c.Equals("/") || c.Equals("#") || c.Equals("^")
				|| c.Equals("%") || c.Equals("~") || c.Equals("@")
				|| c.Equals("$") || c.Equals("!") || c.Equals(">")
				|| c.Equals("<");
		}
		/* In order to parse integers
		 * Check if they begin with a numeric or signed value
		 */
		private bool isNumeric(string c)
		{
			return c.StartsWith("0") || c.StartsWith("1") || c.StartsWith("2")
				|| c.StartsWith("3") || c.StartsWith("4") || c.StartsWith("5")
				|| c.StartsWith("6") || c.StartsWith("7") || c.StartsWith("8")
				|| c.StartsWith("9") || (c.StartsWith("-") && c.Length > 1 && isNumeric(c.Substring(1))); //last check is for negative numbers
		}
		//gives a priority to operations, power is the highest, then multiplication/division, followed by addition and subtraction, then finally all boolean operators
		private int getPriority(string c)
			{
			if (c.Equals("+") || c.Equals("-"))
				return 1;
			else if (c.Equals("*") || c.Equals("/"))
				return 2;
			else if (c.Equals("^"))
				return 3;
			else
				return 0;
			}
		//Read Register Call Function
		private int Read(int register)
		{
			return connection.SendReadCommand((ushort)register);
		}
		//Write Register Call Function
		private int Write(int register,int value)
		{
			return connection.SendWriteCommand((ushort)register, (ushort)value);
		}
		//Delay Function, this will delay the main thread which code is currently running on
		private void Delay(int milliseconds)
		{
			Thread.Sleep(milliseconds);
		}
		//Print Functions call the output bridge print function
		private void Print(object obj)
		{
			outputBridge.PrintOutput(obj.ToString());
		}
		private void Print(string printText, object[] printStatements)
		{
			outputBridge.PrintOutput(printText);
			foreach (object state in printStatements) outputBridge.PrintOutput(state.ToString());
		}
		//PrintLine, Print but with a newline call at the end
		private void PrintLine(object obj)
		{
			outputBridge.PrintLineOutput(obj.ToString());
		}
		private void PrintLine(string printText, object[] printStatements)
		{
			outputBridge.PrintLineOutput(printText);
			foreach (object state in printStatements) outputBridge.PrintLineOutput(state.ToString());
		}



		private int VerifyClosedProgram(List<(int,string)> tokens)
		{
			//This method exists to verify that each brace or parentheses is closed
			//If there is an invalid end this method returns false and the line 
			//number of the invalid end
			


			//Uses a stack to keep track

			Stack<string> bracketStack = new Stack<string>();
			int lastNum = 0;
			foreach((int num, string elem) in tokens)
			{
				lastNum = num;
				if(elem.Equals("{") || elem.Equals("("))
				{
					bracketStack.Push(elem);
				}
				//if error the line number will be outputed
				if (elem.Equals("}"))
				{
					if(bracketStack.Count == 0)
					{
						return num;
					}
					if (!bracketStack.Pop().Equals("{"))
					{
						return num;
					}
				}
				if (elem.Equals(")"))
				{
					if (bracketStack.Count == 0)
					{
						return num;
					}
					if (!bracketStack.Pop().Equals("("))
					{
						return num;
					}
				}
			}
			if(bracketStack.Count != 0)
			{
				return lastNum;
			}
			return -1; //Negative One is the return value for a correct program
		}





	}
