using UnityEngine;

// Perching trajectory: stable initial position, moves first to predefined perpendicular start
public class PerchingTrajectoryV2Modified : MonoBehaviour
{
    [Header("Targets")]
    public Transform targetA;
    public Transform targetB;

    [Header("Start Position Parameters")]
    public float perpendicularDistance = 10f;
    public float startHeight = 6f;

    [Header("Trajectory Parameters")]
    public float heightOffset = 0.3f;
    public float moveVelocity = 2f;
    public float stopDistance = 0.5f;

    [Header("Stabilization")]
    public float pauseDuration = 5f;

    private float startTime;
    private bool isAtStartPosition = false;
    private bool isStabilized = false;
    private bool hasReachedTarget = false;

    private Vector3 perpendicularStartPos;
    private Vector3 finalDestination;

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

        // Calculate perpendicular start position
        Vector2 A2 = new Vector2(pA.x, pA.z);
        Vector2 B2 = new Vector2(pB.x, pB.z);
        Vector2 mid2D = (A2 + B2) * 0.5f;
        Vector2 perpDir = new Vector2(-(B2 - A2).y, (B2 - A2).x).normalized;
        Vector2 start2D = mid2D + perpDir * perpendicularDistance;

        perpendicularStartPos = new Vector3(start2D.x, mid3D.y + startHeight, start2D.y);
        finalDestination = mid3D + Vector3.up * heightOffset;

        startTime = Time.time;
    }

    void FixedUpdate()
    {
        float step = moveVelocity * Time.fixedDeltaTime;

        if (!isAtStartPosition)
        {
            transform.position = Vector3.MoveTowards(transform.position, perpendicularStartPos, step);

            if (Vector3.Distance(transform.position, perpendicularStartPos) <= stopDistance)
            {
                isAtStartPosition = true;
                startTime = Time.time; // reset stabilization timer
            }
            return;
        }

        if (!isStabilized)
        {
            if (Time.time - startTime >= pauseDuration)
                isStabilized = true;
            return;
        }

        if (hasReachedTarget) return;

        transform.position = Vector3.MoveTowards(transform.position, finalDestination, step);

        if (Vector3.Distance(transform.position, finalDestination) <= stopDistance)
        {
            hasReachedTarget = true;
            Debug.Log("Midpoint reached.");
        }
    }
}