using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
	/* Math Operation handles all the operations
	* they are static methods so that no reference to the class is needed
	*/
	public class MathOperations
	{

		/*
		 * Custom Exception for Invalid Operations
		 * Contains the two values being operated on
		 */
		public class OperationException : Exception
		{
			public object value1 { get; set; }
			public object value2 { get; set; }
			public OperationException() : base() { }
			public OperationException(string message) : base(message) { }
			public OperationException(string message, object val1 = null, object val2 = null) : this(message)
			{
				value1 = val1;
				value2 = val2;
			}
		}
		//Checks if values are equal, not the objects themselves
		public static object EqualsCondition(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a == (int)b;
				}
				else if (b is double)
				{
					return (int)a == (double)b;
				}
				else
				{
					return a != b;
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a == (int)b;
				}
				else if (b is double)
				{
					return (double)a == (double)b;
				}
				else
				{
					return a != b;
				}
			}
			else if (a is string)
			{
				if (b is string)
				{
					return ((string)a).Equals((string)b);
				}
				else
					return a == b;
			}
			else if (a is bool)
			{
				if (b is bool)
				{
					return ((bool)a) == (bool)b;
				}
				else
					return a == b;
			}
			else
				return a == b;
		}
		public static object NotEqualsCondition(object a, object b)
		{ /*Original Code, but its redundant since we have the equals method
		   * 
			if (a is int)
			{
				if (b is int)
				{
					return (int)a != (int)b;
				}
				else if (b is double)
				{
					return (int)a != (double)b;
				}
				else
				{
					return a != b;
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a != (int)b;
				}
				else if (b is double)
				{
					return (double)a != (double)b;
				}
				else
				{
					return a != b;
				}
			}
			else if (a is string)
			{
				if (b is string)
				{
					return !((string)a).Equals((string)b);
				}
				else
					return a != b;
			}
			else if (a is bool)
			{
				if (b is bool)
				{
					return ((bool)a) != (bool)b;
				}
				else
					return a != b;
			}
			else
				return a != b;
			*/
			return !(bool)EqualsCondition(a, b);
		}
		//Greater than is a numeric comparision
		//Therefore datatypes doesnt need to be the same
		public static object GreaterThan(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a > (int)b;
				}
				else if (b is double)
				{
					return (int)a > (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Greater Than Comparision", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a > (int)b;
				}
				else if (b is double)
				{
					return (double)a > (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Greater Than Comparision", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Greater Than Comparision", a, b);
			}
		}
		//The code for greater than or equals is almost exactly the same as greater than
		public static object GreaterThanOrEquals(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a >= (int)b;
				}
				else if (b is double)
				{
					return (int)a >= (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Greater Than Or Equals Comparision", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a >= (int)b;
				}
				else if (b is double)
				{
					return (double)a >= (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Greater Than or Equals Comparision", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Greater Than or Equals Comparision", a, b);
			}
		}
		//Less than is also the same kind of method as Greater than, just < instead of >
		public static object LessThan(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a < (int)b;
				}
				else if (b is double)
				{
					return (int)a < (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Less Than Comparision", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a < (int)b;
				}
				else if (b is double)
				{
					return (double)a < (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Less Than Comparision", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Less Than Comparision", a, b);
			}
		}

		public static object LessThanOrEquals(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a <= (int)b;
				}
				else if (b is double)
				{
					return (int)a <= (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Less Than Or Equals Comparision", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a <= (int)b;
				}
				else if (b is double)
				{
					return (double)a <= (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Less Than or Equals Comparision", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Less Than or Equals Comparision", a, b);
			}
		}

		//Add is mostly for addition of numeric types
		//But also it allows for concatenation of strings with other datatypes
		public static object add(object a, object b)
		{
			if (a is string || b is string)
			{
				return a.ToString() + b.ToString();
			}
			else if (a is int)
			{
				if (b is int)
				{
					return (int)a + (int)b;
				}
				else if (b is double)
				{
					return (int)a + (double)b;
				}
				else
				{
					throw new OperationException("Numeric or String Type Required For Addition", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a + (int)b;
				}
				else if (b is double)
				{
					return (double)a + (double)b;
				}
				else
				{
					throw new OperationException("Numeric or String Type Required For Addition", a, b);
				}
			}
			else
			{
				throw new Exception("Numeric or String Type Required For Addition");
			}
		}
		//Subtraction is similar to additon, but there is no support for strings
		public static object subtract(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a - (int)b;
				}
				else if (b is double)
				{
					return (int)a - (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Subtraction", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a - (int)b;
				}
				else if (b is double)
				{
					return (double)a - (double)b;
				}
				else
				{
					throw new Exception("Numeric Type Required For Subtraction");
				}
			}
			else
			{
				throw new Exception("Numeric Type Required For Subtraction");
			}
		}
		//multiply method for numeric types only, uses the same rules for multiplication as c#
		public static object multiply(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a * (int)b;
				}
				else if (b is double)
				{
					return (int)a * (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Multiplication", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a * (int)b;
				}
				else if (b is double)
				{
					return (double)a * (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Multiplication", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Multiplication", a, b);
			}
		}
		//divide method for numeric types only, uses the same rules for multiplication as c#
		public static object divide(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return (int)a / (int)b;
				}
				else if (b is double)
				{
					return (int)a / (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Division", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return (double)a / (int)b;
				}
				else if (b is double)
				{
					return (double)a / (double)b;
				}
				else
				{
					throw new OperationException("Numeric Type Required For Division", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Division", a, b);
			}
		}
		//Power methods are almost never brought into languages, but rather extensions of a math class
		//Since I am coding this using a programming language, well i can use their method
		//Again for numeric types only
		public static object power(object a, object b)
		{
			if (a is int)
			{
				if (b is int)
				{
					return Math.Pow((int)a, (int)b);
				}
				else if (b is double)
				{
					return Math.Pow((int)a, (double)b);
				}
				else
				{
					throw new OperationException("Numeric Type Required For Exponential Operation", a, b);
				}
			}
			else if (a is double)
			{
				if (b is int)
				{
					return Math.Pow((double)a, (int)b);
				}
				else if (b is double)
				{
					return Math.Pow((double)a, (double)b);
				}
				else
				{
					throw new OperationException("Numeric Type Required For Exponential Operation", a, b);
				}
			}
			else
			{
				throw new OperationException("Numeric Type Required For Exponential Operation", a, b);
			}
		}
	}
