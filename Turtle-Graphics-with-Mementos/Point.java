public class Point {
	//Point class represent a 2D Grid
		private double x;
		private double y;
		public Point() {
			this.x = 0;
			this.y = 0;
		}
		public Point(double x, double y) {
			this.x = x;
			this.y = y;
		}
		//getters and setter for x and y
		public void setX(double x) {
			this.x = x;
		}
		public void setY(double y) {
			this.y = y;
		}
		public void setXY(double x, double y) {
			setX(x);
			setY(y);
		}
		public double getX() {
			return rounded(x,4);
		}
		public double getY() {
			return rounded(y,4);
		}
		private double rounded(double val,int decimalNum) {
			double tenFactor = Math.pow(10.0, decimalNum);
			int intVal = (int)Math.round(val*tenFactor);
			return intVal/tenFactor;
		}
		//moveX and moveY add to the current value
		public void moveX(double xChange) {
			x += xChange;
		}
		public void moveY(double yChange) {
			y += yChange;
		}
		public void moveXY(double xChange, double yChange) {
			moveX(xChange);
			moveY(yChange);
		}
		//clone method so the to ensure the location isn't changed when Turtle is copied
		public Point clone() {
			return new Point(x,y);
		}
		public String toString() {
			return getX()+" "+getY();
		}
}