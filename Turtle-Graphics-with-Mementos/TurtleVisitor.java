import java.util.*;

public class TurtleVisitor extends Visitor {
	//holds list of Memento
	private ArrayList<Memento> mementos;
	public TurtleVisitor() {
		mementos = new ArrayList<Memento>();
	}
	public ArrayList<Memento> getStates(){
		return mementos;
	}
	//visit adds the AbstractExpressions state to list of memento
	public void visit(AssignmentExpression assignExpression) {
		mementos.add(assignExpression.getMemento());
	}
	public void visit(MoveExpression moveExpression) {
		mementos.add(moveExpression.getMemento());
	}
	public void visit(TurnExpression turnExpression) {
		mementos.add(turnExpression.getMemento());
	}
	public void visit(PenDownExpression penDownExpression) {
		mementos.add(penDownExpression.getMemento());
	}
	public void visit(PenUpExpression penUpExpression) {
		mementos.add(penUpExpression.getMemento());
	}
	public void visit(RepeatExpression repeatExpression) {
		//Not sure if really needed even
		mementos.add(repeatExpression.getMemento());
	}

}
