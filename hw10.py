
float angle = 0;          
float aVelocity = 0.05;   
float amplitude = 120;    

float originX = 400;      
float originY = 200;      
float boxSize = 60;       
float wallX = 100;        

void setup() {
  size(800, 400);         
  rectMode(CENTER);       
  strokeCap(ROUND);       
}

void draw() {
  background(220);
  
  float x = originX + sin(angle) * amplitude;
  
  angle += aVelocity;
  
  stroke(255);           
  strokeWeight(4);        
  line(wallX, originY - 60, wallX, originY + 60);   
  
  float springStartX = wallX; 
  
  float springEndX = x - boxSize/2;
  
  noFill();
  stroke(255);            
  strokeWeight(3);        
  
  int segments = 24;     
  float springLength = springEndX - springStartX;
  float segmentLen = springLength / segments;
  
  beginShape();

  vertex(springStartX, originY);
  
  for (int i = 1; i < segments; i++) {
    float px = springStartX + i * segmentLen;
    float py = originY;
    
    if (i % 2 == 0) {
      py -= 15; 
    } else {
      py += 15; 
    }
    
    vertex(px, py);
  }
  
  vertex(springEndX, originY);
  endShape();

  noStroke();
  fill(0, 0, 255);        
  rect(x, originY, boxSize, boxSize);
}
