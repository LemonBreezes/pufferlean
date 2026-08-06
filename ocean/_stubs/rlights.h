#ifndef RLIGHTS_H
#define RLIGHTS_H
#define MAX_LIGHTS 4
#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1
typedef struct { int type; int enabled; Vector3 position; Vector3 target; Color color; float attenuation;
  int enabledLoc, typeLoc, positionLoc, targetLoc, colorLoc, attenuationLoc; } Light;
static inline Light CreateLight(int type, Vector3 position, Vector3 target, Color color, Shader shader){
  (void)shader; Light l; l.type=type; l.enabled=1; l.position=position; l.target=target; l.color=color; l.attenuation=0; return l; }
static inline void UpdateLightValues(Shader shader, Light light){ (void)shader; (void)light; }
#endif
