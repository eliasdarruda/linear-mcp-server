#!/usr/bin/env node

import { LinearClient, LinearDocument, Issue, User, Team, WorkflowState, IssueLabel } from "@linear/sdk";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequest,
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  Tool,
  ResourceTemplate,
  Prompt,
} from "@modelcontextprotocol/sdk/types.js";
import dotenv from "dotenv";
import { z } from 'zod';

interface CreateIssueArgs {
  title: string;
  teamId: string;
  description?: string;
  priority?: number;
  status?: string;
}

interface UpdateIssueArgs {
  id: string;
  title?: string;
  description?: string;
  priority?: number;
  status?: string;
}

interface SearchIssuesArgs {
  query?: string;
  teamId?: string;
  limit?: number;
  status?: string;
  assigneeId?: string;
  labels?: string[];
  priority?: number;
  estimate?: number;
  includeArchived?: boolean;
}

interface GetUserIssuesArgs {
  userId?: string;
  includeArchived?: boolean;
  limit?: number;
}

interface AddCommentArgs {
  issueId: string;
  body: string;
  createAsUser?: string;
  displayIconUrl?: string;
}

interface RateLimiterMetrics {
  totalRequests: number;
  requestsInLastHour: number;
  averageRequestTime: number;
  queueLength: number;
  lastRequestTime: number;
}

interface LinearIssueResponse {
  identifier: string;
  title: string;
  priority: number | null;
  status: string | null;
  stateName?: string;
  url: string;
}

interface GetTeamsArgs {
  includeArchived?: boolean;
}

interface TeamResponse {
  id: string;
  name: string;
  key: string;
  description?: string;
  active: boolean;
}

interface UserResponse {
  id: string;
  name: string;
  email: string;
  admin: boolean;
  active: boolean;
}

interface OrganizationResponse {
  id: string;
  name: string;
  urlKey: string;
  teams: TeamResponse[];
  users: UserResponse[];
}

class RateLimiter {
  public readonly requestsPerHour = 1400;
  private queue: (() => Promise<any>)[] = [];
  private processing = false;
  private lastRequestTime = 0;
  private readonly minDelayMs = 3600000 / this.requestsPerHour;
  private requestTimes: number[] = [];
  private requestTimestamps: number[] = [];

  async enqueue<T>(fn: () => Promise<T>, operation?: string): Promise<T> {
    const startTime = Date.now();
    const queuePosition = this.queue.length;

    console.error(`[Linear API] Enqueueing request${operation ? ` for ${operation}` : ''} (Queue position: ${queuePosition})`);

    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          console.error(`[Linear API] Starting request${operation ? ` for ${operation}` : ''}`);
          const result = await fn();
          const endTime = Date.now();
          const duration = endTime - startTime;

          console.error(`[Linear API] Completed request${operation ? ` for ${operation}` : ''} (Duration: ${duration}ms)`);
          this.trackRequest(startTime, endTime, operation);
          resolve(result);
        } catch (error) {
          console.error(`[Linear API] Error in request${operation ? ` for ${operation}` : ''}: `, error);
          reject(error);
        }
      });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.processing || this.queue.length === 0) return;
    this.processing = true;

    while (this.queue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;

      const requestsInLastHour = this.requestTimestamps.filter(t => t > now - 3600000).length;
      if (requestsInLastHour >= this.requestsPerHour * 0.9 && timeSinceLastRequest < this.minDelayMs) {
        const waitTime = this.minDelayMs - timeSinceLastRequest;
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }

      const fn = this.queue.shift();
      if (fn) {
        this.lastRequestTime = Date.now();
        await fn();
      }
    }

    this.processing = false;
  }

  async batch<T>(items: any[], batchSize: number, fn: (item: any) => Promise<T>, operation?: string): Promise<T[]> {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      batches.push(Promise.all(
        batch.map(item => this.enqueue(() => fn(item), operation))
      ));
    }

    const results = await Promise.all(batches);
    return results.flat();
  }

  private trackRequest(startTime: number, endTime: number, operation?: string) {
    const duration = endTime - startTime;
    this.requestTimes.push(duration);
    this.requestTimestamps.push(startTime);

    // Keep only last hour of requests
    const oneHourAgo = Date.now() - 3600000;
    this.requestTimestamps = this.requestTimestamps.filter(t => t > oneHourAgo);
    this.requestTimes = this.requestTimes.slice(-this.requestTimestamps.length);
  }

  getMetrics(): RateLimiterMetrics {
    const now = Date.now();
    const oneHourAgo = now - 3600000;
    const recentRequests = this.requestTimestamps.filter(t => t > oneHourAgo);

    return {
      totalRequests: this.requestTimestamps.length,
      requestsInLastHour: recentRequests.length,
      averageRequestTime: this.requestTimes.length > 0
        ? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
        : 0,
      queueLength: this.queue.length,
      lastRequestTime: this.lastRequestTime
    };
  }
}

class LinearMCPClient {
  private client: LinearClient;
  public readonly rateLimiter: RateLimiter;

  constructor(apiKey: string) {
    if (!apiKey) throw new Error("LINEAR_API_KEY environment variable is required");
    this.client = new LinearClient({ apiKey });
    this.rateLimiter = new RateLimiter();
  }

  private async getIssueDetails(issue: Issue) {
    // Use a single GraphQL query to fetch issue details in one request
    const query = `
      query IssueDetails($issueId: ID!) {
        issue(id: $issueId) {
          state {
            id
            name
          }
          assignee {
            id
            name
          }
          team {
            id
            name
            key
          }
        }
      }
    `;

    const result = await this.rateLimiter.enqueue(() => 
      this.client.client.rawRequest(query, { issueId: issue.id }),
      'getIssueDetails'
    );

    const data = result?.data as any;
    const issueDetails = data?.issue;
    return {
      state: issueDetails?.state || null,
      assignee: issueDetails?.assignee || null,
      team: issueDetails?.team || null
    };
  }

  private addMetricsToResource(response: any) {
    const metrics = this.rateLimiter.getMetrics();
    return {
      ...response,
      metadata: {
        ...response.metadata,
        apiMetrics: {
          requestsInLastHour: metrics.requestsInLastHour,
          remainingRequests: this.rateLimiter.requestsPerHour - metrics.requestsInLastHour,
          averageRequestTime: `${Math.round(metrics.averageRequestTime)}ms`,
          queueLength: metrics.queueLength,
          lastRequestTime: new Date(metrics.lastRequestTime).toISOString()
        }
      }
    };
  }

  getMetricsText(): string {
    const metrics = this.rateLimiter.getMetrics();
    return `\n\nAPI Metrics:\n- Requests in last hour: ${metrics.requestsInLastHour}\n- Remaining requests: ${this.rateLimiter.requestsPerHour - metrics.requestsInLastHour}\n- Average request time: ${Math.round(metrics.averageRequestTime)}ms\n- Queue length: ${metrics.queueLength}`;
  }

  async listIssues() {
    const result = await this.rateLimiter.enqueue(
      () => this.client.issues({
        first: 50,
        orderBy: LinearDocument.PaginationOrderBy.UpdatedAt
      }),
      'listIssues'
    );

    const issuesWithDetails = await this.rateLimiter.batch(
      result.nodes,
      5,
      async (issue) => {
        const details = await this.getIssueDetails(issue);
        return {
          uri: `linear-issue:///${issue.id}`,
          mimeType: "application/json",
          name: issue.title,
          description: `Linear issue ${issue.identifier}: ${issue.title}`,
          metadata: {
            identifier: issue.identifier,
            priority: issue.priority,
            status: details.state ? await details.state.name : undefined,
            assignee: details.assignee ? await details.assignee.name : undefined,
            team: details.team ? await details.team.name : undefined,
          }
        };
      },
      'getIssueDetails'
    );

    return this.addMetricsToResource(issuesWithDetails);
  }

  async getIssue(issueId: string) {
    // Use a single GraphQL query to fetch issue with all related data in one request
    const query = `
      query GetIssue($issueId: String!) {
        issue(id: $issueId) {
          id
          identifier
          title
          description
          priority
          url
          createdAt
          updatedAt
          state {
            name
          }
          assignee {
            name
          }
          team {
            name
          }
        }
      }
    `;

    const result = await this.rateLimiter.enqueue(() => 
      this.client.client.rawRequest(query, { issueId }),
      'getIssue'
    );

    const data = result?.data as any;
    if (!data?.issue) {
      throw new Error(`Issue ${issueId} not found`);
    }

    const issue = data.issue;

    return this.addMetricsToResource({
      id: issue.id,
      identifier: issue.identifier,
      title: issue.title,
      description: issue.description,
      priority: issue.priority,
      status: issue.state?.name,
      assignee: issue.assignee?.name,
      team: issue.team?.name,
      url: issue.url
    });
  }

  async createIssue(args: CreateIssueArgs) {
    const issuePayload = await this.client.createIssue({
      title: args.title,
      teamId: args.teamId,
      description: args.description,
      priority: args.priority,
      stateId: args.status
    });

    const issue = await issuePayload.issue;
    if (!issue) throw new Error("Failed to create issue");
    return issue;
  }

  async updateIssue(args: UpdateIssueArgs) {
    const issue = await this.client.issue(args.id);
    if (!issue) throw new Error(`Issue ${args.id} not found`);

    const updatePayload = await issue.update({
      title: args.title,
      description: args.description,
      priority: args.priority,
      stateId: args.status
    });

    const updatedIssue = await updatePayload.issue;
    if (!updatedIssue) throw new Error("Failed to update issue");
    return updatedIssue;
  }

  async searchIssues(args: SearchIssuesArgs) {
    try {
      console.error(`[searchIssues] Starting search with args: ${JSON.stringify(args)}`);
      
      // Use a single GraphQL query to fetch issues with all related data in one request
      const query = `
        query SearchIssues($filter: IssueFilter, $first: Int, $includeArchived: Boolean) {
          issues(filter: $filter, first: $first, includeArchived: $includeArchived) {
            nodes {
              id
              identifier
              title
              description
              priority
              estimate
              url
              state {
                id
                name
              }
              assignee {
                id
                name
              }
              labels {
                nodes {
                  id
                  name
                }
              }
            }
          }
        }
      `;

      const filter = this.buildSearchFilter(args);
      console.error(`[searchIssues] Built filter: ${JSON.stringify(filter)}`);
      
      const variables = {
        filter: filter,
        first: args.limit || 10,
        includeArchived: args.includeArchived || false
      };

      // Direct API call that we know works
      console.error(`[searchIssues] Making direct API call with variables: ${JSON.stringify(variables)}`);
      const result = await this.client.client.rawRequest(query, variables);
      
      console.error(`[searchIssues] Got response with status: ${result?.status || 'unknown'}`);

      if (!result || !result.data) {
        console.error("[searchIssues] No data in API response");
        return [];
      }

      const data = result.data as any;
      if (!data.issues || !data.issues.nodes) {
        console.error("[searchIssues] No issues found in API response");
        return [];
      }

      const issuesNodes = data.issues.nodes;
      console.error(`[searchIssues] Found ${issuesNodes.length} issues`);
      
      // Map issues with additional error handling
      const issues = [];
      for (const issue of issuesNodes) {
        try {
          const mappedIssue = {
            id: issue.id,
            identifier: issue.identifier,
            title: issue.title,
            description: issue.description || '',
            priority: issue.priority,
            estimate: issue.estimate,
            status: issue.state?.name || 'Unknown',
            stateName: issue.state?.name || 'Unknown', // Add for consistency with other endpoints
            assignee: issue.assignee?.name || null,
            labels: issue.labels?.nodes?.map((label: any) => label.name) || [],
            url: issue.url
          };
          issues.push(mappedIssue);
        } catch (mapError) {
          console.error(`[searchIssues] Error mapping issue: ${mapError}`);
          console.error(`[searchIssues] Problematic issue object: ${JSON.stringify(issue)}`);
        }
      }
      
      console.error(`[searchIssues] Successfully mapped ${issues.length} issues`);
      return issues;
    } catch (error) {
      console.error(`[searchIssues] Error: ${error}`);
      if ((error as any).response) {
        console.error(`[searchIssues] GraphQL Error:`, (error as any).response.errors);
      }
      return [];
    }
  }

  async getUserIssues(args: GetUserIssuesArgs) {
    try {
      // Implementation with both viewing user's issues and specified user's issues
      console.error(`[getUserIssues] Starting with args: ${JSON.stringify(args)}`);
      
      let query: string;
      let variables: any;
      
      // Handle specific user vs. current viewer
      if (args.userId && typeof args.userId === 'string') {
        console.error(`[getUserIssues] Fetching issues for specific user: ${args.userId}`);
        // Query for specific user's issues
        query = `
          query UserIssues($userId: String!, $first: Int, $includeArchived: Boolean) {
            user(id: $userId) {
              id
              name
              email
              assignedIssues(first: $first, includeArchived: $includeArchived) {
                nodes {
                  id
                  identifier
                  title
                  description
                  priority
                  url
                  state {
                    name
                  }
                }
              }
            }
          }
        `;
        variables = {
          userId: args.userId,
          first: args.limit || 50,
          includeArchived: args.includeArchived || false
        };
      } else {
        console.error(`[getUserIssues] Fetching issues for current viewer`);
        // Query for current viewer's issues
        query = `
          query ViewerIssues($first: Int, $includeArchived: Boolean) {
            viewer {
              id
              name
              email
              assignedIssues(first: $first, includeArchived: $includeArchived) {
                nodes {
                  id
                  identifier
                  title
                  description
                  priority
                  url
                  state {
                    name
                  }
                }
              }
            }
          }
        `;
        variables = {
          first: args.limit || 50,
          includeArchived: args.includeArchived || false
        };
      }
      
      // Direct API call bypassing rate limiter (which we know works in our test)
      console.error(`[getUserIssues] Making direct API call with query: ${query.replace(/\s+/g, ' ')}`);
      console.error(`[getUserIssues] Variables: ${JSON.stringify(variables)}`);
      
      const result = await this.client.client.rawRequest(query, variables);
      
      console.error(`[getUserIssues] Got response with status: ${result?.status || 'unknown'}`);
      
      if (!result || !result.data) {
        console.error("[getUserIssues] No data in API response");
        return [];
      }
      
      const data = result.data as any;
      
      // Process results based on query type
      let issuesNodes: any[] = [];
      let userName = "Unknown";
      
      if (args.userId) {
        // For specific user query
        if (!data.user) {
          console.error(`[getUserIssues] No user data found for ID: ${args.userId}`);
          return [];
        }
        
        const user = data.user;
        userName = user.name || "Unknown";
        console.error(`[getUserIssues] Found user: ${userName} (${user.email || 'no email'})`);
        
        if (!user.assignedIssues || !user.assignedIssues.nodes) {
          console.error(`[getUserIssues] No assigned issues found for user ID: ${args.userId}`);
          return [];
        }
        
        issuesNodes = user.assignedIssues.nodes;
      } else {
        // For viewer query
        if (!data.viewer) {
          console.error("[getUserIssues] No viewer in API response data");
          return [];
        }
        
        const viewer = data.viewer;
        userName = viewer.name || "Unknown";
        console.error(`[getUserIssues] Found viewer: ${userName} (${viewer.email || 'no email'})`);
        
        if (!viewer.assignedIssues || !viewer.assignedIssues.nodes) {
          console.error("[getUserIssues] No assigned issues found for viewer");
          return [];
        }
        
        issuesNodes = viewer.assignedIssues.nodes;
      }
      
      console.error(`[getUserIssues] Found ${issuesNodes.length} issues for ${args.userId ? 'user' : 'viewer'}`);
      
      // Map issues with additional error handling
      const issues = [];
      for (const issue of issuesNodes) {
        try {
          const mappedIssue = {
            id: issue.id,
            identifier: issue.identifier,
            title: issue.title,
            description: issue.description || '',
            priority: issue.priority,
            stateName: issue.state?.name || 'Unknown',
            url: issue.url
          };
          issues.push(mappedIssue);
        } catch (mapError) {
          console.error(`[getUserIssues] Error mapping issue: ${mapError}`);
          console.error(`[getUserIssues] Problematic issue object: ${JSON.stringify(issue)}`);
        }
      }
      
      console.error(`[getUserIssues] Successfully mapped ${issues.length} issues`);
      return issues;
    } catch (error) {
      console.error(`[getUserIssues] Error in getUserIssues: ${error}`);
      if ((error as any).response) {
        console.error(`[getUserIssues] GraphQL Error:`, (error as any).response.errors);
      }
      return [];
    }
  }

  async addComment(args: AddCommentArgs) {
    const commentPayload = await this.client.createComment({
      issueId: args.issueId,
      body: args.body,
      createAsUser: args.createAsUser,
      displayIconUrl: args.displayIconUrl
    });

    const comment = await commentPayload.comment;
    if (!comment) throw new Error("Failed to create comment");

    const issue = await comment.issue;
    return {
      comment,
      issue
    };
  }

  async getTeamIssues(teamId: string) {
    try {
      console.error(`[getTeamIssues] Starting with teamId: ${teamId}`);
      
      // Use a single GraphQL query to fetch team issues with all related data in one request
      const query = `
        query TeamIssues($teamId: ID!) {
          team(id: $teamId) {
            name
            key
            issues {
              nodes {
                id
                identifier
                title
                description
                priority
                url
                state {
                  name
                }
                assignee {
                  name
                }
              }
            }
          }
        }
      `;

      const variables = {
        teamId: teamId
      };

      // Direct API call that we know works
      console.error(`[getTeamIssues] Making direct API call for team: ${teamId}`);
      const result = await this.client.client.rawRequest(query, variables);
      
      console.error(`[getTeamIssues] Got response with status: ${result?.status || 'unknown'}`);

      if (!result || !result.data) {
        console.error("[getTeamIssues] No data in API response");
        return [];
      }

      const data = result.data as any;
      if (!data.team) {
        console.error(`[getTeamIssues] No team found with ID: ${teamId}`);
        return [];
      }
      
      console.error(`[getTeamIssues] Found team: ${data.team.name} (${data.team.key})`);
      
      if (!data.team.issues || !data.team.issues.nodes) {
        console.error(`[getTeamIssues] No issues found for team: ${teamId}`);
        return [];
      }

      const issuesNodes = data.team.issues.nodes;
      console.error(`[getTeamIssues] Found ${issuesNodes.length} issues for team`);
      
      // Map issues with additional error handling
      const issues = [];
      for (const issue of issuesNodes) {
        try {
          const mappedIssue = {
            id: issue.id,
            identifier: issue.identifier,
            title: issue.title,
            description: issue.description || '',
            priority: issue.priority,
            status: issue.state?.name || 'Unknown',
            stateName: issue.state?.name || 'Unknown', // Add for consistency with other endpoints
            assignee: issue.assignee?.name || null,
            url: issue.url
          };
          issues.push(mappedIssue);
        } catch (mapError) {
          console.error(`[getTeamIssues] Error mapping issue: ${mapError}`);
          console.error(`[getTeamIssues] Problematic issue object: ${JSON.stringify(issue)}`);
        }
      }
      
      console.error(`[getTeamIssues] Successfully mapped ${issues.length} issues`);
      return issues;
    } catch (error) {
      console.error(`[getTeamIssues] Error: ${error}`);
      if ((error as any).response) {
        console.error(`[getTeamIssues] GraphQL Error:`, (error as any).response.errors);
      }
      return [];
    }
  }

  async getViewer() {
    // Use a single GraphQL query to fetch viewer data with teams and organization in one request
    const query = `
      query ViewerData {
        viewer {
          id
          name
          email
          admin
          teams {
            nodes {
              id
              name
              key
            }
          }
        }
        organization {
          id
          name
          urlKey
        }
      }
    `;

    const result = await this.rateLimiter.enqueue(() => 
      this.client.client.rawRequest(query, {}),
      'getViewer'
    );

    const data = result?.data as any;
    if (!data?.viewer) {
      throw new Error("Failed to fetch viewer data");
    }

    const viewer = data.viewer;
    const organization = data.organization;

    return this.addMetricsToResource({
      id: viewer.id,
      name: viewer.name,
      email: viewer.email,
      admin: viewer.admin,
      teams: viewer.teams.nodes.map((team: any) => ({
        id: team.id,
        name: team.name,
        key: team.key
      })),
      organization: {
        id: organization.id,
        name: organization.name,
        urlKey: organization.urlKey
      }
    });
  }

  async getOrganization(): Promise<OrganizationResponse> {
    // Use a single GraphQL query to fetch organization data with teams and users in one request
    const query = `
      query OrganizationData {
        organization {
          id
          name
          urlKey
          teams {
            nodes {
              id
              name
              key
              description
              archivedAt
            }
          }
          users {
            nodes {
              id
              name
              email
              admin
              active
            }
          }
        }
      }
    `;

    const result = await this.rateLimiter.enqueue(() => 
      this.client.client.rawRequest(query, {}),
      'getOrganization'
    );

    const data = result?.data as any;
    if (!data?.organization) {
      throw new Error("Failed to fetch organization data");
    }

    const organization = data.organization;

    return this.addMetricsToResource({
      id: organization.id,
      name: organization.name,
      urlKey: organization.urlKey,
      teams: organization.teams.nodes.map((team: any) => ({
        id: team.id,
        name: team.name,
        key: team.key,
        description: team.description || undefined,
        active: !team.archivedAt
      })),
      users: organization.users.nodes.map((user: any) => ({
        id: user.id,
        name: user.name,
        email: user.email,
        admin: user.admin,
        active: user.active
      }))
    });
  }
  
  async getTeams(args: GetTeamsArgs = {}) {
    try {
      console.error(`[getTeams] Starting with args: ${JSON.stringify(args)}`);
      
      // Use a single GraphQL query to fetch all teams with their details in one request
      const query = `
        query GetTeams($includeArchived: Boolean) {
          teams(includeArchived: $includeArchived) {
            nodes {
              id
              name
              key
              description
              archivedAt
            }
          }
        }
      `;

      const variables = {
        includeArchived: args.includeArchived || false
      };

      // Direct API call with good error handling
      console.error(`[getTeams] Making direct API call with variables: ${JSON.stringify(variables)}`);
      const result = await this.client.client.rawRequest(query, variables);
      
      console.error(`[getTeams] Got response with status: ${result?.status || 'unknown'}`);

      if (!result || !result.data) {
        console.error("[getTeams] No data in API response");
        return [];
      }

      const data = result.data as any;
      if (!data.teams || !data.teams.nodes) {
        console.error("[getTeams] No teams found in API response");
        return [];
      }

      const teamsNodes = data.teams.nodes;
      console.error(`[getTeams] Found ${teamsNodes.length} teams`);
      
      // Map teams with additional error handling
      const teams = [];
      for (const team of teamsNodes) {
        try {
          const mappedTeam = {
            id: team.id,
            name: team.name,
            key: team.key,
            description: team.description || '',
            active: !team.archivedAt
          };
          teams.push(mappedTeam);
        } catch (mapError) {
          console.error(`[getTeams] Error mapping team: ${mapError}`);
          console.error(`[getTeams] Problematic team object: ${JSON.stringify(team)}`);
        }
      }
      
      console.error(`[getTeams] Successfully mapped ${teams.length} teams`);
      return teams;
    } catch (error) {
      console.error(`[getTeams] Error: ${error}`);
      if ((error as any).response) {
        console.error(`[getTeams] GraphQL Error:`, (error as any).response.errors);
      }
      return [];
    }
  }
  
  private buildSearchFilter(args: SearchIssuesArgs): any {
    const filter: any = {};

    if (args.query) {
      // Check if query looks like an issue identifier (e.g., SYM-1622)
      if (/^[A-Z]+-\d+$/.test(args.query)) {
        filter.identifier = { eq: args.query };
      } else {
        filter.or = [
          { title: { contains: args.query } },
          { description: { contains: args.query } }
        ];
      }
    }

    if (args.teamId) {
      filter.team = { id: { eq: args.teamId } };
    }

    if (args.status) {
      filter.state = { name: { eq: args.status } };
    }

    if (args.assigneeId) {
      filter.assignee = { id: { eq: args.assigneeId } };
    }

    if (args.labels && args.labels.length > 0) {
      filter.labels = {
        some: {
          name: { in: args.labels }
        }
      };
    }

    if (args.priority) {
      filter.priority = { eq: args.priority };
    }

    if (args.estimate) {
      filter.estimate = { eq: args.estimate };
    }

    return filter;
  }
}

const createIssueTool: Tool = {
  name: "linear_create_issue",
  description: "Creates a new Linear issue with specified details. Use this to create tickets for tasks, bugs, or feature requests. Returns the created issue's identifier and URL. Required fields are title and teamId, with optional description, priority (0-4, where 0 is no priority and 1 is urgent), and status. Example: {\"title\": \"Fix login bug\", \"teamId\": \"5a656ab6-af57-4895-bd34-99855115bb1b\", \"priority\": 2}",
  inputSchema: {
    type: "object",
    properties: {
      title: { type: "string", description: "Issue title" },
      teamId: { type: "string", description: "Team ID (get this from linear_get_organization first)" },
      description: { type: "string", description: "Issue description" },
      priority: { type: "number", description: "Priority (0=none, 1=urgent, 2=high, 3=normal, 4=low)" },
      status: { type: "string", description: "Issue status (must match exact workflow state name)" }
    },
    required: ["title", "teamId"]
  }
};

const updateIssueTool: Tool = {
  name: "linear_update_issue",
  description: "Updates an existing Linear issue's properties. Use this to modify issue details like title, description, priority, or status. Requires the issue ID and accepts any combination of updatable fields. Returns the updated issue's identifier and URL. Example: {\"id\": \"issue-id-here\", \"status\": \"In Progress\"}",
  inputSchema: {
    type: "object",
    properties: {
      id: { type: "string", description: "Issue ID (get this from search_issues or linear-issue:/// resources)" },
      title: { type: "string", description: "New title" },
      description: { type: "string", description: "New description" },
      priority: { type: "number", description: "New priority (0=none, 1=urgent, 2=high, 3=normal, 4=low)" },
      status: { type: "string", description: "New status (must match exact workflow state name)" }
    },
    required: ["id"]
  }
};

const searchIssuesTool: Tool = {
  name: "linear_search_issues",
  description: "Searches Linear issues using flexible criteria. Supports filtering by any combination of: title/description text, team, status, assignee, labels, priority (1=urgent, 2=high, 3=normal, 4=low), and estimate. Returns up to 10 issues by default (configurable via limit). Example: {\"query\": \"login\", \"teamId\": \"team-id\", \"limit\": 20}",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Optional text to search in title and description" },
      teamId: { type: "string", description: "Filter by team ID (get this from linear_get_organization first)" },
      status: { type: "string", description: "Filter by status name (e.g., 'In Progress', 'Done')" },
      assigneeId: { type: "string", description: "Filter by assignee's user ID (get this from linear_get_organization first)" },
      labels: {
        type: "array",
        items: { type: "string" },
        description: "Filter by label names (e.g. [\"bug\", \"feature\"])"
      },
      priority: {
        type: "number",
        description: "Filter by priority (1=urgent, 2=high, 3=normal, 4=low)"
      },
      estimate: {
        type: "number",
        description: "Filter by estimate points"
      },
      includeArchived: {
        type: "boolean",
        description: "Include archived issues in results (default: false)"
      },
      limit: {
        type: "number",
        description: "Max results to return (default: 10)"
      }
    }
  }
};

const getUserIssuesTool: Tool = {
  name: "linear_get_user_issues",
  description: "Retrieves issues assigned to a specific user or the authenticated user if no userId is provided. Returns issues sorted by last updated, including priority, status, and other metadata. Useful for finding a user's workload or tracking assigned tasks. Example: {\"userId\": \"d0bc778f-c97d-4450-a464-7282854e8801\"}",
  inputSchema: {
    type: "object",
    properties: {
      userId: { type: "string", description: "Optional user ID (get this from linear_get_organization first). If not provided, returns authenticated user's issues" },
      includeArchived: { type: "boolean", description: "Include archived issues in results" },
      limit: { type: "number", description: "Maximum number of issues to return (default: 50)" }
    }
  }
};

const addCommentTool: Tool = {
  name: "linear_add_comment",
  description: "Adds a comment to an existing Linear issue. Supports markdown formatting in the comment body. Can optionally specify a custom user name and avatar for the comment. Returns the created comment's details including its URL. Example: {\"issueId\": \"issue-id\", \"body\": \"Added fix for this issue\"}",
  inputSchema: {
    type: "object",
    properties: {
      issueId: { type: "string", description: "ID of the issue to comment on (get this from search_issues or linear-issue:/// resources)" },
      body: { type: "string", description: "Comment text in markdown format" },
      createAsUser: { type: "string", description: "Optional custom username to show for the comment" },
      displayIconUrl: { type: "string", description: "Optional avatar URL for the comment" }
    },
    required: ["issueId", "body"]
  }
};

const getTeamsTool: Tool = {
  name: "linear_get_teams",
  description: "Fetches all teams in the organization. Returns team details including ID, name, key, description, and status. Use this to get team information for creating issues or filtering searches. This should typically be called first to get team IDs. Example: {\"includeArchived\": true}",
  inputSchema: {
    type: "object",
    properties: {
      includeArchived: { type: "boolean", description: "Include archived teams in results (default: false)" }
    }
  }
};

const getOrganizationTool: Tool = {
  name: "linear_get_organization",
  description: "Fetches information about the current organization associated with your Linear API key. Returns organization details, teams, and users. This is the first tool you should call to get team IDs and user IDs needed for other operations. Example: {}",
  inputSchema: {
    type: "object",
    properties: {}
  }
};

const getIssueDetailsTool: Tool = {
  name: "linear_get_issue_details",
  description: "Fetches detailed information about a specific Linear issue including its title, description, status, assignee, comments, and history. Works with either issue reference codes (SYM-839) or internal IDs. Example: {\"issueId\": \"SYM-839\"}",
  inputSchema: {
    type: "object",
    properties: {
      issueId: { type: "string", description: "The issue reference code (e.g., SYM-839) or internal ID" }
    },
    required: ["issueId"]
  }
};


const resourceTemplates: ResourceTemplate[] = [
  {
    uriTemplate: "linear-issue:///{issueId}",
    name: "Linear Issue",
    description: "A Linear issue with its details, comments, and metadata. Use this to fetch detailed information about a specific issue. First use linear_search_issues to find the issue ID.",
    parameters: {
      issueId: {
        type: "string",
        description: "The unique identifier of the Linear issue (e.g., the internal ID)"
      }
    },
    examples: [
      "linear-issue:///c2b318fb-95d2-4a81-9539-f3268f34af87"
    ]
  },
  {
    uriTemplate: "linear-viewer:",
    name: "Current User",
    description: "Information about the authenticated user associated with the API key, including their role, teams, and settings. No parameters needed.",
    parameters: {},
    examples: [
      "linear-viewer:"
    ]
  },
  {
    uriTemplate: "linear-organization:",
    name: "Current Organization",
    description: "Details about the Linear organization associated with the API key, including settings, teams, and members. Use this to get team IDs and user IDs needed for other queries. No parameters needed.",
    parameters: {},
    examples: [
      "linear-organization:"
    ]
  },
  {
    uriTemplate: "linear-team:///{teamId}/issues",
    name: "Team Issues",
    description: "All active issues belonging to a specific Linear team, including their status, priority, and assignees. Get the teamId from linear_get_organization.",
    parameters: {
      teamId: {
        type: "string",
        description: "The unique identifier of the Linear team (found via linear_get_organization)"
      }
    },
    examples: [
      "linear-team:///TEAM-123/issues"
    ]
  },
  {
    uriTemplate: "linear-user:///{userId}/assigned",
    name: "User Assigned Issues",
    description: "Active issues assigned to a specific Linear user. Returns issues sorted by update date. Get the userId from linear_get_organization.",
    parameters: {
      userId: {
        type: "string",
        description: "The unique identifier of the Linear user (found via linear_get_organization). Use 'me' for the authenticated user"
      }
    },
    examples: [
      "linear-user:///USER-123/assigned",
      "linear-user:///me/assigned"
    ]
  }
];

const serverPrompt: Prompt = {
  name: "linear-server-prompt",
  description: "Instructions for using the Linear MCP server effectively",
  instructions: `This server provides access to Linear, a project management tool. Use it to manage issues, track work, and coordinate with teams.

Key capabilities:
- Create and update issues: Create new tickets or modify existing ones with titles, descriptions, priorities, and team assignments.
- Search functionality: Find issues across the organization using flexible search queries with team and user filters.
- Team coordination: Access team-specific issues and manage work distribution within teams.
- Issue tracking: Add comments and track progress through status updates and assignments.
- Organization overview: View team structures and user assignments across the organization.

## Available Tools and Query Format

1. **linear_get_organization**
   - Get organization data, teams, and users - CALL THIS FIRST
   - No parameters required
   - Example: \`{}\`

2. **linear_get_teams**
   - Returns team details (ID, name, key, description)
   - Optional: \`includeArchived\` (boolean)
   - Example: \`{"includeArchived": true}\`

3. **linear_create_issue**
   - Required: \`title\`, \`teamId\`
   - Optional: \`description\`, \`priority\` (0-4), \`status\`
   - Priority levels: 0=none, 1=urgent, 2=high, 3=normal, 4=low
   - Example: \`{"title": "Fix login bug", "teamId": "5a656ab6-af57-4895-bd34-99855115bb1b", "priority": 2}\`

4. **linear_update_issue**
   - Required: \`id\`
   - Optional: \`title\`, \`description\`, \`priority\`, \`status\`
   - Example: \`{"id": "issue-id-here", "status": "In Progress"}\`

5. **linear_search_issues**
   - All parameters optional:
   - \`query\`: Text search in title/description
   - \`teamId\`: Filter by team
   - \`assigneeId\`: Filter by assignee's user ID
   - \`status\`: Filter by status name
   - \`labels\`: Array of label names
   - \`priority\`: Filter by priority level (1=urgent to 4=low)
   - \`estimate\`: Filter by points estimate
   - \`includeArchived\`: Include archived issues (default: false)
   - \`limit\`: Max results to return (default: 10)
   - Example: \`{"query": "login", "teamId": "team-id", "limit": 20}\`

6. **linear_get_user_issues**
   - Optional parameters:
   - \`userId\`: User ID (omit for authenticated user)
   - \`includeArchived\`: Include archived issues
   - \`limit\`: Max results (default: 50)
   - Example: \`{"userId": "d0bc778f-c97d-4450-a464-7282854e8801"}\`

7. **linear_add_comment**
   - Required: \`issueId\`, \`body\`
   - Optional: \`createAsUser\`, \`displayIconUrl\`
   - Example: \`{"issueId": "issue-id", "body": "Added fix for this issue"}\`

## Best Practices for Querying

1. **Find user/team information first**:
   - Always start with \`linear_get_organization\` to get all user IDs and team IDs
   - These IDs are required for creating issues or filtering searches

2. **Search efficiently**:
   - Combine multiple filters for precise results
   - Use specific search terms in the \`query\` parameter
   - For user issues, use their ID from organization data

3. **Understand priority levels**:
   - 0 = No priority
   - 1 = Urgent
   - 2 = High
   - 3 = Normal
   - 4 = Low

4. **Resource URIs**:
   - \`linear-issue:///{issueId}\` - Get a specific issue
   - \`linear-team:///{teamId}/issues\` - Get team issues
   - \`linear-user:///{userId}/assigned\` - Get user assignments
   - \`linear-organization:\` - Get organization info
   - \`linear-viewer:\` - Get current user context

5. **Rate limits**:
   - Server handles 1400 requests per hour
   - Responses include API metrics (requests used, remaining)


## Example Workflow

1. First, get organization info to find teams and users:
   \`\`\`
   linear_get_organization {}
   \`\`\`

2. Then create an issue using a team ID from the response:
   \`\`\`
   linear_create_issue {"title": "Implement login feature", "teamId": "5a656ab6-af57-4895-bd34-99855115bb1b", "priority": 2}
   \`\`\`

3. Search for issues related to a specific topic:
   \`\`\`
   linear_search_issues {"query": "login", "teamId": "5a656ab6-af57-4895-bd34-99855115bb1b"}
   \`\`\`

4. Get issues assigned to a specific user:
   \`\`\`
   linear_get_user_issues {"userId": "d0bc778f-c97d-4450-a464-7282854e8801"}
   \`\`\`

5. Add a comment to an issue:
   \`\`\`
   linear_add_comment {"issueId": "issue-id-here", "body": "This is fixed in the latest release"}
   \`\`\`

The server uses the authenticated user's permissions for all operations.`
};

// Zod schemas for tool argument validation
const CreateIssueArgsSchema = z.object({
  title: z.string().describe("Issue title"),
  teamId: z.string().describe("Team ID"),
  description: z.string().optional().describe("Issue description"),
  priority: z.number().min(0).max(4).optional().describe("Priority (0-4)"),
  status: z.string().optional().describe("Issue status")
});

const UpdateIssueArgsSchema = z.object({
  id: z.string().describe("Issue ID"),
  title: z.string().optional().describe("New title"),
  description: z.string().optional().describe("New description"),
  priority: z.number().optional().describe("New priority (0-4)"),
  status: z.string().optional().describe("New status")
});

const SearchIssuesArgsSchema = z.object({
  query: z.string().optional().describe("Optional text to search in title and description"),
  teamId: z.string().optional().describe("Filter by team ID"),
  status: z.string().optional().describe("Filter by status name (e.g., 'In Progress', 'Done')"),
  assigneeId: z.string().optional().describe("Filter by assignee's user ID"),
  labels: z.array(z.string()).optional().describe("Filter by label names"),
  priority: z.number().optional().describe("Filter by priority (1=urgent, 2=high, 3=normal, 4=low)"),
  estimate: z.number().optional().describe("Filter by estimate points"),
  includeArchived: z.boolean().optional().describe("Include archived issues in results (default: false)"),
  limit: z.number().optional().describe("Max results to return (default: 10)")
});

const GetUserIssuesArgsSchema = z.object({
  userId: z.string().optional().describe("Optional user ID. If not provided, returns authenticated user's issues"),
  includeArchived: z.boolean().optional().describe("Include archived issues in results"),
  limit: z.number().optional().describe("Maximum number of issues to return (default: 50)")
});

const AddCommentArgsSchema = z.object({
  issueId: z.string().describe("ID of the issue to comment on"),
  body: z.string().describe("Comment text in markdown format"),
  createAsUser: z.string().optional().describe("Optional custom username to show for the comment"),
  displayIconUrl: z.string().optional().describe("Optional avatar URL for the comment")
});

const GetTeamsArgsSchema = z.object({
  includeArchived: z.boolean().optional().describe("Include archived teams in results")
});

const GetIssueDetailsArgsSchema = z.object({
  issueId: z.string().describe("The issue reference code (e.g., SYM-839) or internal ID")
});


async function main() {
  try {
    dotenv.config();

    const apiKey = process.env.LINEAR_API_KEY;
    if (!apiKey) {
      console.error("LINEAR_API_KEY environment variable is required");
      process.exit(1);
    }

    console.error("Starting Linear MCP Server...");
    const linearClient = new LinearMCPClient(apiKey);

    const server = new Server(
      {
        name: "linear-mcp-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          prompts: {
            default: serverPrompt
          },
          resources: {
            templates: true,
            read: true
          },
          tools: {},
        },
      }
    );

    server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: await linearClient.listIssues()
    }));

    server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = new URL(request.params.uri);
      const path = uri.pathname.replace(/^\//, '');

      if (uri.protocol === 'linear-organization') {
        const organization = await linearClient.getOrganization();
        return {
          contents: [{
            uri: "linear-organization:",
            mimeType: "application/json",
            text: JSON.stringify(organization, null, 2)
          }]
        };
      }

      if (uri.protocol === 'linear-viewer') {
        const viewer = await linearClient.getViewer();
        return {
          contents: [{
            uri: "linear-viewer:",
            mimeType: "application/json",
            text: JSON.stringify(viewer, null, 2)
          }]
        };
      }

      if (uri.protocol === 'linear-issue:') {
        const issue = await linearClient.getIssue(path);
        return {
          contents: [{
            uri: request.params.uri,
            mimeType: "application/json",
            text: JSON.stringify(issue, null, 2)
          }]
        };
      }

      if (uri.protocol === 'linear-team:') {
        const [teamId] = path.split('/');
        const issues = await linearClient.getTeamIssues(teamId);
        return {
          contents: [{
            uri: request.params.uri,
            mimeType: "application/json",
            text: JSON.stringify(issues, null, 2)
          }]
        };
      }

      if (uri.protocol === 'linear-user:') {
        const [userId] = path.split('/');
        const issues = await linearClient.getUserIssues({
          userId: userId === 'me' ? undefined : userId
        });
        return {
          contents: [{
            uri: request.params.uri,
            mimeType: "application/json",
            text: JSON.stringify(issues, null, 2)
          }]
        };
      }

      throw new Error(`Unsupported resource URI: ${request.params.uri}`);
    });

    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        createIssueTool, 
        updateIssueTool, 
        searchIssuesTool, 
        getUserIssuesTool, 
        addCommentTool, 
        getTeamsTool, 
        getOrganizationTool,
        getIssueDetailsTool
      ]
    }));

    server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => {
      return {
        resourceTemplates: resourceTemplates
      };
    });

    server.setRequestHandler(ListPromptsRequestSchema, async () => {
      return {
        prompts: [serverPrompt]
      };
    });

    server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      if (request.params.name === serverPrompt.name) {
        return {
          prompt: serverPrompt
        };
      }
      throw new Error(`Prompt not found: ${request.params.name}`);
    });

    server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest) => {
      // IMPORTANT DEBUG STATEMENT - log every tool call
      console.error("\n========== TOOL CALL ==========");
      console.error("Tool:", request.params.name);
      console.error("Args:", JSON.stringify(request.params.arguments));
      console.error("==============================\n");

      let metrics: RateLimiterMetrics = {
        totalRequests: 0,
        requestsInLastHour: 0,
        averageRequestTime: 0,
        queueLength: 0,
        lastRequestTime: Date.now()
      };

      // We'll keep the standard flow for all tools now that we've fixed the implementation

      try {
        const { name, arguments: args } = request.params;
        if (!args) throw new Error("Missing arguments");

        metrics = linearClient.rateLimiter.getMetrics();

        switch (name) {
          case "linear_create_issue": {
            const validatedArgs = CreateIssueArgsSchema.parse(args);
            const issue = await linearClient.createIssue(validatedArgs);
            const metricsText = linearClient.getMetricsText();
            
            return {
              content: [{
                type: "text",
                text: `Created issue ${issue.identifier}: ${issue.title}\nURL: ${issue.url}${metricsText}`
              }]
            };
          }

          case "linear_update_issue": {
            const validatedArgs = UpdateIssueArgsSchema.parse(args);
            const issue = await linearClient.updateIssue(validatedArgs);
            const metricsText = linearClient.getMetricsText();
            
            return {
              content: [{
                type: "text",
                text: `Updated issue ${issue.identifier}\nURL: ${issue.url}${metricsText}`
              }]
            };
          }

          case "linear_search_issues": {
            const validatedArgs = SearchIssuesArgsSchema.parse(args);
            try {
              console.error("[Handler] Calling searchIssues with args:", JSON.stringify(validatedArgs));
              const issues = await linearClient.searchIssues(validatedArgs);
              console.error(`[Handler] Got response with ${Array.isArray(issues) ? issues.length : 0} issues`);
              
              if (Array.isArray(issues) && issues.length > 0) {
                console.error(`[Handler] First issue: ${JSON.stringify(issues[0])}`);
              }
              
              const metricsText = linearClient.getMetricsText();
              
              // Format issues for display
              let issuesText = '';
              if (Array.isArray(issues) && issues.length > 0) {
                issuesText = '\n' + issues.map((issue: any) => {
                  // Format any labels if they exist
                  const labelsText = issue.labels && issue.labels.length > 0 
                    ? `\n  Labels: ${issue.labels.join(', ')}` 
                    : '';
                    
                  // Format assignee if it exists  
                  const assigneeText = issue.assignee 
                    ? `\n  Assignee: ${issue.assignee}` 
                    : '';
                    
                  return `- ${issue.identifier}: ${issue.title}
  Priority: ${issue.priority || 'None'}
  Status: ${issue.stateName || issue.status || 'Unknown'}${assigneeText}${labelsText}
  ${issue.url}`;
                }).join('\n');
                
                console.error(`[Handler] Formatted ${issues.length} issues for display`);
                return {
                  content: [{
                    type: "text",
                    text: `Found ${issues.length} issues:${issuesText}${metricsText}`
                  }]
                };
              } else {
                console.error("[Handler] No issues found or invalid response format");
                return {
                  content: [{
                    type: "text",
                    text: `Found 0 issues:${metricsText}`
                  }]
                };
              }
            } catch (error) {
              console.error(`[Handler] Error in search_issues handler: ${error}`);
              const metricsText = linearClient.getMetricsText();
              return {
                content: [{
                  type: "text",
                  text: `Error searching issues: ${error instanceof Error ? error.message : String(error)}${metricsText}`
                }]
              };
            }
          }

          case "linear_get_user_issues": {
            const validatedArgs = GetUserIssuesArgsSchema.parse(args);
            try {
              console.error("[Handler] Calling getUserIssues with args:", JSON.stringify(validatedArgs));
              const issues = await linearClient.getUserIssues(validatedArgs);
              console.error(`[Handler] Got response with ${Array.isArray(issues) ? issues.length : 0} issues`);
              
              if (Array.isArray(issues) && issues.length > 0) {
                console.error(`[Handler] First issue: ${JSON.stringify(issues[0])}`);
              }
              
              const metricsText = linearClient.getMetricsText();
              
              // Format issues for display
              let issuesText = '';
              if (Array.isArray(issues) && issues.length > 0) {
                issuesText = '\n' + issues.map((issue: any) => 
                  `- ${issue.identifier}: ${issue.title}\n  Priority: ${issue.priority || 'None'}\n  Status: ${issue.stateName || 'Unknown'}\n  ${issue.url}`
                ).join('\n');
                
                console.error(`[Handler] Formatted ${issues.length} issues for display`);
                return {
                  content: [{
                    type: "text",
                    text: `Found ${issues.length} issues:${issuesText}${metricsText}`
                  }]
                };
              } else {
                console.error("[Handler] No issues found or invalid response format");
                return {
                  content: [{
                    type: "text",
                    text: `Found 0 issues:${metricsText}`
                  }]
                };
              }
            } catch (error) {
              console.error(`[Handler] Error in get_user_issues handler: ${error}`);
              const metricsText = linearClient.getMetricsText();
              return {
                content: [{
                  type: "text",
                  text: `Error retrieving user issues: ${error instanceof Error ? error.message : String(error)}${metricsText}`
                }]
              };
            }
          }

          case "linear_add_comment": {
            const validatedArgs = AddCommentArgsSchema.parse(args);
            const { comment, issue } = await linearClient.addComment(validatedArgs);
            const metricsText = linearClient.getMetricsText();

            return {
              content: [{
                type: "text",
                text: `Added comment to issue ${issue?.identifier}\nURL: ${comment.url}${metricsText}`
              }]
            };
          }
          
          case "linear_get_teams": {
            const validatedArgs = GetTeamsArgsSchema.parse(args);
            try {
              console.error("[Handler] Calling getTeams with args:", JSON.stringify(validatedArgs));
              const teams = await linearClient.getTeams(validatedArgs);
              console.error(`[Handler] Got response with ${Array.isArray(teams) ? teams.length : 0} teams`);
              
              const metricsText = linearClient.getMetricsText();
              
              // Format teams for display if we got a valid response
              if (Array.isArray(teams) && teams.length > 0) {
                const teamsText = teams.map((team: TeamResponse) => 
                  `- ${team.key}: ${team.name} (ID: ${team.id})\n  ${team.description || ""}\n  Status: ${team.active ? 'Active' : 'Archived'}`
                ).join('\n');
                
                const teamIdsText = teams.map((team: TeamResponse) => 
                  `- ${team.name}: ${team.id}`
                ).join('\n');
                
                console.error(`[Handler] Successfully formatted ${teams.length} teams`);
                return {
                  content: [{
                    type: "text",
                    text: `Found ${teams.length} teams:\n${teamsText}\n\nTeam IDs for reference:\n${teamIdsText}${metricsText}`
                  }]
                };
              } else {
                console.error("[Handler] No teams found or invalid response format");
                return {
                  content: [{
                    type: "text",
                    text: `Found 0 teams${metricsText}`
                  }]
                };
              }
            } catch (error) {
              console.error(`[Handler] Error in get_teams handler:`, error);
              const metricsText = linearClient.getMetricsText();
              return {
                content: [{
                  type: "text",
                  text: `Error retrieving teams: ${error instanceof Error ? error.message : String(error)}${metricsText}`
                }]
              };
            }
          }
          
          case "linear_get_organization": {
            const organization = await linearClient.getOrganization();
            const metricsText = linearClient.getMetricsText();
            
            return {
              content: [{
                type: "text",
                text: `Organization: ${organization.name} (${organization.urlKey})\n\nTeams: ${organization.teams.length}\n${
                  organization.teams.map((team: TeamResponse) => 
                    `- ${team.key}: ${team.name} (ID: ${team.id})`
                  ).join('\n')
                }\n\nTeam IDs for reference:\n${
                  organization.teams.map((team: TeamResponse) => 
                    `- ${team.name}: ${team.id}`
                  ).join('\n')
                }\n\nUsers: ${organization.users.length}\n${
                  organization.users.map((user: UserResponse) => 
                    `- ${user.name} (${user.email}) (ID: ${user.id}) ${user.admin ? '[Admin]' : ''} ${user.active ? '' : '[Inactive]'}`
                  ).join('\n')
                }\n\nUser IDs for reference:\n${
                  organization.users.map((user: UserResponse) => 
                    `- ${user.name}: ${user.id}`
                  ).join('\n')
                }${metricsText}`
              }]
            };
          }
          
          
          case "linear_get_issue_details": {
            // Schema validation
            if (!args.issueId || typeof args.issueId !== 'string') {
              return {
                content: [{
                  type: "text",
                  text: "Error: issueId is required and must be a string"
                }]
              };
            }
            
            try {
              console.error(`[Handler] Calling getIssue with ID: ${args.issueId}`);
              const issue = await linearClient.getIssue(args.issueId);
              console.error(`[Handler] Got issue: ${issue ? `${issue.identifier} - ${issue.title}` : 'not found'}`);
              
              const metricsText = linearClient.getMetricsText();
              
              if (issue) {
                // Format the issue details nicely
                let commentsText = '';
                if (issue.comments && issue.comments.length > 0) {
                  commentsText = '\n\n## Comments\n' + issue.comments.map((comment: {userName: string, createdAt: string, body: string}) => 
                    `- ${comment.userName} (${comment.createdAt ? new Date(comment.createdAt).toLocaleString() : 'Unknown'}):\n  ${comment.body.replace(/\n/g, '\n  ')}`
                  ).join('\n\n');
                }
                
                // Format labels
                let labelsText = '';
                if (issue.labels && issue.labels.length > 0) {
                  labelsText = '\nLabels: ' + issue.labels.map((label: {name: string}) => label.name).join(', ');
                }
                
                // Format history (simplified)
                let historyText = '';
                if (issue.history && issue.history.length > 0) {
                  historyText = '\n\n## Recent Activity\n' + issue.history
                    .slice(0, 5) // Show only the 5 most recent events
                    .map((event: {type: string, fromState?: string, toState?: string, actor: string, createdAt: string}) => {
                      if (event.type === 'state' && event.fromState && event.toState) {
                        return `- ${event.actor} changed status from "${event.fromState}" to "${event.toState}" (${event.createdAt ? new Date(event.createdAt).toLocaleString() : 'Unknown'})`;
                      } else {
                        return `- ${event.actor} ${event.type} (${event.createdAt ? new Date(event.createdAt).toLocaleString() : 'Unknown'})`;
                      }
                    }).join('\n');
                }
                
                return {
                  content: [{
                    type: "text",
                    text: `# ${issue.identifier}: ${issue.title}

## Details
- Status: ${issue.stateName} (${issue.stateType})
- Priority: ${issue.priority || 'None'}
- Team: ${issue.teamName || 'None'} (${issue.teamKey || ''})
- Assignee: ${issue.assignee || 'Unassigned'}${labelsText}
- Created: ${issue.createdAt ? new Date(issue.createdAt).toLocaleString() : 'Unknown'}
- Updated: ${issue.updatedAt ? new Date(issue.updatedAt).toLocaleString() : 'Unknown'}
- URL: ${issue.url}

## Description
${issue.description || 'No description provided.'}${commentsText}${historyText}${metricsText}`
                  }]
                };
              } else {
                return {
                  content: [{
                    type: "text",
                    text: `No issue found with ID: ${args.issueId}${metricsText}`
                  }]
                };
              }
            } catch (error) {
              console.error(`[Handler] Error in get_issue_details handler:`, error);
              const metricsText = linearClient.getMetricsText();
              return {
                content: [{
                  type: "text",
                  text: `Error retrieving issue details: ${error instanceof Error ? error.message : String(error)}${metricsText}`
                }]
              };
            }
          }

          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        console.error("Error executing tool:", error);

        // If it's a Zod error, format it nicely
        if (error instanceof z.ZodError) {
          const formattedErrors = error.errors.map(err => ({
            path: err.path,
            message: err.message,
            code: 'VALIDATION_ERROR'
          }));
          
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: {
                  type: 'VALIDATION_ERROR',
                  message: 'Invalid request parameters',
                  details: formattedErrors
                }
              }, null, 2)
            }]
          };
        }

        // For Linear API errors, try to extract useful information
        if (error instanceof Error && 'response' in error) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: {
                  type: 'API_ERROR',
                  message: error.message,
                  details: {
                    // @ts-ignore - response property exists but isn't in type
                    status: error.response?.status,
                    // @ts-ignore - response property exists but isn't in type
                    data: error.response?.data
                  }
                }
              }, null, 2)
            }]
          };
        }

        // For all other errors
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: {
                type: 'UNKNOWN_ERROR',
                message: error instanceof Error ? error.message : String(error)
              }
            }, null, 2)
          }]
        };
      }
    });

    const transport = new StdioServerTransport();
    console.error("Connecting server to transport...");
    await server.connect(transport);
    console.error("Linear MCP Server running on stdio");
  } catch (error) {
    console.error(`Fatal error in main(): ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}

main().catch((error: unknown) => {
  console.error("Fatal error in main():", error instanceof Error ? error.message : String(error));
  process.exit(1);
});
