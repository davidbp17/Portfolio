
public class PenDownExpression extends AbstractExpression{
	//PenDownExpression changes turtle to penDown state
	private Memento state;
	public PenDownExpression() {
	}
	public Memento getMemento() {
		return state;
	}
	public void evaluate(Context context,Turtle turtle) {
		//calls turtle.penDown() and saves state
		turtle.penDown();
		state = turtle.createMemento();
	}
	public void accept(Visitor visitor) {
		visitor.visit(this);
	}
	
	public PenDownExpression copy() {
		return new PenDownExpression();
	}
}