using UnityEngine;

// Perching trajectory: Tau-based curved dive (α–coupling version)
// Original Version 
// Probably everything working, k, kd, curved angle

public class PerchingTrajectoryV4 : MonoBehaviour
{
    [Header("Targets")]
    public Transform targetA;
    public Transform targetB;

    [Header("Start Position Parameters")]
    public float perpendicularDistance = 10f;
    public float startHeight = 6f;

    [Header("Tau Trajectory Parameters")]
    public float initialVelocity = 5f;
    public float tauShapeParamk = 0.4f;
    public float kdAlpha = 0.4f;
    public int numTrajectoryPoints = 5000;
    public float heightOffset = 0.3f;
    public float stopDistance = 0.1f;

    [Header("Stabilization")]
    public float pauseDuration = 5f;

    private Vector3 perpendicularStartPos;
    private Vector3 finalDestination;
    private Vector3[] tauTrajectory;

    private float startTime;
    private bool isAtStartPosition = false;
    private bool isStabilized = false;
    private bool hasReachedTarget = false;

    void Start()
    {
        if (targetA == null || targetB == null)
        {
            Debug.LogError("Assign both targetA and targetB.");
            enabled = false;
            return;
        }

        // 1) compute start and touchdown positions
        Vector3 pA = targetA.position;
        Vector3 pB = targetB.position;
        Vector3 mid3D = (pA + pB) * 0.5f;
        Vector2 A2 = new Vector2(pA.x, pA.z), B2 = new Vector2(pB.x, pB.z);
        Vector2 mid2D = (A2 + B2) * 0.5f;
        Vector2 perpDir = new Vector2(-(B2 - A2).y, (B2 - A2).x).normalized;
        Vector2 start2D = mid2D + perpDir * perpendicularDistance;

        perpendicularStartPos = new Vector3(start2D.x, mid3D.y + startHeight, start2D.y);
        finalDestination = mid3D + Vector3.up * heightOffset;

        // 2) precompute curved tau‐law trajectory
        GenerateTauTrajectory(perpendicularStartPos, finalDestination);
        startTime = Time.time;
    }

    void FixedUpdate()
    {
        // Move to start perch
        if (!isAtStartPosition)
        {
            float step = initialVelocity * Time.fixedDeltaTime;
            transform.position = Vector3.MoveTowards(transform.position, perpendicularStartPos, step);

            if ((transform.position - perpendicularStartPos).sqrMagnitude <= stopDistance * stopDistance)
            {
                isAtStartPosition = true;
                startTime = Time.time;
            }
            return;
        }

        // Stabilize hover
        if (!isStabilized)
        {
            if (Time.time - startTime >= pauseDuration)
            {
                isStabilized = true;
                startTime = Time.time;
            }
            return;
        }

        // Follow curved trajectory
        if (hasReachedTarget || tauTrajectory == null) return;

        float elapsed = Time.time - startTime;
        float duration = numTrajectoryPoints * Time.fixedDeltaTime;
        float tNorm = Mathf.Clamp01(elapsed / duration);
        int idx = Mathf.Min(Mathf.FloorToInt(tNorm * (numTrajectoryPoints - 1)), numTrajectoryPoints - 1);

        transform.position = tauTrajectory[idx];
        if (idx >= numTrajectoryPoints - 1)
        {
            hasReachedTarget = true;
            Debug.Log("Touchdown reached.");
        }
    }

    void GenerateTauTrajectory(Vector3 p0, Vector3 p_td)
    {
        float d0 = Vector3.Distance(p0, p_td);
        Vector3 delta = p0 - p_td;

        // unit direction in XZ (ground plane)
        Vector3 dirXZ = new Vector3(delta.x, 0f, delta.z).normalized;

        // initial pitch angle α₀ between vertical and d₀
        float initialVertGap = p0.y - p_td.y;
        float alpha0 = Mathf.Asin(initialVertGap / d0);

        // tau‐law parameters
        float tau0 = -d0 / initialVelocity;
        float t_d = -tau0 / tauShapeParamk;
        float invK = 1f / tauShapeParamk;
        float invKd = 1f / kdAlpha;

        tauTrajectory = new Vector3[numTrajectoryPoints];
        for (int i = 0; i < numTrajectoryPoints; i++)
        {
            // time and distance gap
            float t = (t_d * i) / (numTrajectoryPoints - 1);
            float d = d0 * Mathf.Pow(1f - t / t_d, invK);

            // α‐coupling
            float alpha = alpha0 * Mathf.Pow(d / d0, invKd);
            float cosA = Mathf.Cos(alpha);
            float sinA = Mathf.Sin(alpha);

            // horizontal reach & vertical rise
            float h = d * cosA;
            float y = p_td.y + d * sinA;

            // build curved point
            Vector3 horizDisp = dirXZ * h;
            tauTrajectory[i] = new Vector3(
                p_td.x + horizDisp.x,
                y,
                p_td.z + horizDisp.z
            );
        }
    }
}
