import java.util.*;
public class Context {
	//holds list of variables
	public ArrayList<IntegerVariable> varList;
	//Integer Variable class just holds string and integer
	public class IntegerVariable{
		public String varName;
		public Integer intVal;
		public IntegerVariable(String varName, Integer intVal) {
			this.varName = varName;
			this.intVal = intVal;
		}
		
	};
	public Context() {
		varList = new ArrayList<IntegerVariable>();
	}
	//lookup returns value if the variable exists
	//returns null if it doesn't
	public Integer lookup(String varName) {
		for(IntegerVariable var: varList)
		{
			if(var.varName.equals(varName))
				return var.intVal;
		}
		return null;
	}
	//assign checks if variable has already been assigned and changes it
	//adds another if not
	public void assign(String varName, Integer intVal) {
		for(IntegerVariable var: varList)
		{
			if(var.varName.equals(varName)) {
				var.intVal = intVal;
				return;
			}
		}
		varList.add(new IntegerVariable(varName,intVal));
	}
}
