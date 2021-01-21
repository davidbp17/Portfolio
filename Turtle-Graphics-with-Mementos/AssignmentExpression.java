
public class AssignmentExpression extends AbstractExpression {
	//Class to assign a variable in the context
	private String varName;
	private Constant value;
	private Memento state;
	public AssignmentExpression(String varName,Constant constant) {
		//Require a variable name and a value
		this.varName = varName;
		value = constant;
	}
	public void evaluate(Context context,Turtle turtle) {
		//adds variable to context and creates a memento of the current state
		context.assign(varName, value.evaluate(context));
		state = turtle.createMemento();
	}
	public Memento getMemento() {
		return state;
	}
	public void accept(Visitor visitor) {
		visitor.visit(this);
	}
	public AssignmentExpression copy() {
		return new AssignmentExpression(varName,value);
	}
}
