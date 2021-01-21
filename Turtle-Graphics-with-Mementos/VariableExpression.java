
public class VariableExpression extends IntegerExpression {
	//Expression for if variable
	private String varName;
	public VariableExpression(String varName) {
		this.varName = varName;
	}
	//Evaluates the value if called, null pointer exception if doesn't exist
	public Integer evaluate(Context context) {
		if(context == null)
			throw new NullPointerException();
		return context.lookup(varName);
	}
	public IntegerExpression copy() {
		return new VariableExpression(varName);
	}

}
