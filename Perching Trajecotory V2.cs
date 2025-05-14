using UnityEngine;

// Perching trajectory: Tau-based trajectory implementation
public class PerchingTrajectoryV2: MonoBehaviour
{
    [Header("Targets")]
    public Transform targetA;
    public Transform targetB;

    [Header("Start Position Parameters")]
    public float perpendicularDistance = 10f;
    public float startHeight = 6f;

    [Header("Tau Trajectory Parameters")]
    public float initialVelocity = 5f;
    public float tauShapeParam = 0.1f;
    public int numTrajectoryPoints = 100;
    public float heightOffset = 0.3f;
    public float stopDistance = 0.1f;

    [Header("Stabilization")]
    public float pauseDuration = 5f;

    private float startTime;
    private bool isAtStartPosition = false;
    private bool isStabilized = false;
    private bool hasReachedTarget = false;

    private Vector3 perpendicularStartPos;
    private Vector3 finalDestination;
    private Vector3[] tauTrajectory;
    private int currentTrajectoryIndex = 0;

    void Start()
    {
        if (targetA == null || targetB == null)
        {
            Debug.LogError("Assign both targetA and targetB.");
            enabled = false;
            return;
        }

        Vector3 pA = targetA.position;
        Vector3 pB = targetB.position;
        Vector3 mid3D = (pA + pB) * 0.5f;

        Vector2 A2 = new Vector2(pA.x, pA.z);
        Vector2 B2 = new Vector2(pB.x, pB.z);
        Vector2 mid2D = (A2 + B2) * 0.5f;
        Vector2 perpDir = new Vector2(-(B2 - A2).y, (B2 - A2).x).normalized;
        Vector2 start2D = mid2D + perpDir * perpendicularDistance;

        perpendicularStartPos = new Vector3(start2D.x, mid3D.y + startHeight, start2D.y);
        finalDestination = mid3D + Vector3.up * heightOffset;

        GenerateTauTrajectory(perpendicularStartPos, finalDestination);

        startTime = Time.time;
    }

    void FixedUpdate()
    {
        float step;

        Debug.Log("Before Aligning at Midpoint");

        if (!isAtStartPosition)
        {
            step = initialVelocity * Time.fixedDeltaTime;
            transform.position = Vector3.MoveTowards(transform.position, perpendicularStartPos, step);

            if (Vector3.Distance(transform.position, perpendicularStartPos) <= stopDistance)
            {
                isAtStartPosition = true;
                startTime = Time.time;
            }

            Debug.Log("Aligning at Midpoint");

            return;
        }

        if (!isStabilized)
        {
            if (Time.time - startTime >= pauseDuration)
                isStabilized = true;
            
            Debug.Log("Stablizing");
            return;
        }

        if (hasReachedTarget || tauTrajectory == null) return;

        step = initialVelocity * Time.fixedDeltaTime;
        transform.position = Vector3.MoveTowards(transform.position, tauTrajectory[currentTrajectoryIndex], step);

        if (Vector3.Distance(transform.position, tauTrajectory[currentTrajectoryIndex]) <= stopDistance)
        {
            currentTrajectoryIndex++;
            if (currentTrajectoryIndex >= tauTrajectory.Length)
            {
                hasReachedTarget = true;
                Debug.Log("Midpoint reached.");
            }
        }

        Debug.Log("Traying to follow Tau Trajectory");
    }

    void GenerateTauTrajectory(Vector3 p0, Vector3 p_td)
    {
        float d0 = Vector3.Distance(p0, p_td);
        float tau0 = -d0 / initialVelocity;
        float t_d = -tau0 / tauShapeParam;

        tauTrajectory = new Vector3[numTrajectoryPoints];
        for (int i = 0; i < numTrajectoryPoints; i++)
        {
            float t = (t_d / (numTrajectoryPoints - 1)) * i;
            float d = d0 * Mathf.Pow(1 - t / t_d, 1.0f / tauShapeParam);
            float lerpFactor = 1 - (d / d0);
            tauTrajectory[i] = Vector3.Lerp(p0, p_td, lerpFactor);
        }
    }
}
