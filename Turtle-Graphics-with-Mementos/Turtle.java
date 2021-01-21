
public class Turtle implements Memento{
	private boolean penLifted;
	//location of turtle
	private Point point;
	private int degrees;
	public Turtle() {
		point = new Point();
		penLifted = true;
		degrees = 0;
	}
	//moves the turtle a given amount based on direction and distance
	public void move(int distance) {
		double degrees = Math.toRadians(this.degrees);
		double xDistance = (Math.cos(degrees)*distance);
		double yDistance = (Math.sin(degrees)*distance);
		point.moveXY(xDistance, yDistance);
	}
	//turns the turtle
	public void turn(int degrees) {
		this.degrees = (this.degrees+degrees)%360;
	}
	public void penUp() {
		penLifted = true;
	}
	public void penDown() {
		penLifted = false;
	}
	public boolean isPenUp() {
		return penLifted;
	}
	public int direction() {
		return degrees;
	}
	public Point getLocation() {
		return point;
	}
	public void setLocation(Point point) {
		this.point = point;
	}
	//Creates memento of current turtle state
	public Memento createMemento() {
		Memento state;
		try {
			state = (Memento)this.clone();
		}
		catch(CloneNotSupportedException ex) {
			//null if not supported, never will be run
			state = null;
		}
		return state;
	}
	//Restores a turtle to a given state
	public void restoreState(Memento prevState) {
		Turtle newState = (Turtle)prevState;
		setLocation(newState.getLocation());
		if(newState.isPenUp())
			penUp();
		else
			penDown();
		degrees = newState.direction();
	}
	//Clone method
	public Turtle clone() throws CloneNotSupportedException {
		Turtle cloneTurtle = new Turtle();
		if(!penLifted)
			cloneTurtle.penDown();
		cloneTurtle.turn(degrees);
		cloneTurtle.setLocation(point.clone());
		return cloneTurtle;
	}
}

